use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex, RwLock,
    },
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use eframe::egui::{self, CentralPanel, Color32, Grid, RichText, SidePanel, Slider, Vec2b, Window};
use egui_plot::{Arrows, Line, Plot, PlotBounds, PlotPoints};

use glam::Vec2;
use itertools::Itertools;
use midi_msg::{ChannelVoiceMsg, ControlChange, MidiMsg};
use midir::MidiInput;
use rayon::iter::{ParallelBridge, ParallelIterator};
use ringbuf::traits::{Consumer, Producer, Split};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    Fft, FftPlanner,
};
use serde::{Deserialize, Serialize};

const VIS_BUF_SECS: f32 = 4.0;
const VIS_BUF_SIZE: usize = (44100.0 * VIS_BUF_SECS) as usize;
const VIS_BUF_SLACK: usize = 44100 * 3;

const STRINGS_COUNT: usize = 128;
const LOWEST_KEY: usize = 21;
const DAMPING_MUTED: f32 = 0.0003;
const DAMPING_OPEN: f32 = 0.00002;
// simulation runs this many steps per sample
const SUB_STEPS: usize = 3;

trait Lerp {
    fn lerp(&self, percent: f32) -> f32;
}

impl Lerp for [f32; 2] {
    fn lerp(&self, percent: f32) -> f32 {
        self[0] + (self[1] - self[0]) * percent
    }
}

impl Lerp for [u32; 2] {
    fn lerp(&self, percent: f32) -> f32 {
        self[0] as f32 + (self[1] as f32 - self[0] as f32) * percent
    }
}

#[derive(Clone)]
struct Piano {
    strings: Vec<PianoString>,
    sustain: bool,
    pluck_force: f32,
    vocoder_level: f32,
}

#[derive(Clone)]
struct PianoString {
    conf: StringConfig,
    state: StringState,
}

#[derive(Clone, Default)]
struct StringState {
    damping: f32,
    /// key is pressed
    on: bool,
    /// damper is removed by a key or by a sustain pedal
    is_active: bool,
    /// will be simulated
    is_ringing: bool,
    current_frequency: f32,
    pos: Vec<Vec2>,
    vel: Vec<Vec2>,
    ran_away: bool,
}

#[derive(Serialize, Deserialize)]
struct SerializedPiano {
    strings: Vec<StringConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StringConfig {
    boost: f32,
    size: u32,
    tension: f32,
    inertia: f32,
    elasticity: f32,
    expected_frequency: f32,
}

impl From<StringConfig> for PianoString {
    fn from(value: StringConfig) -> Self {
        PianoString {
            state: StringState {
                damping: DAMPING_MUTED,
                on: false,
                is_active: false,
                is_ringing: false,
                current_frequency: 0.0,
                ran_away: false,
                pos: (0..value.size)
                    .map(|i| Vec2 {
                        x: i as f32,
                        y: 0.0,
                    })
                    .collect(),
                vel: vec![Vec2::ZERO; value.size as usize],
            },
            conf: value,
        }
    }
}

impl From<&PianoString> for StringConfig {
    fn from(value: &PianoString) -> Self {
        value.conf.clone()
    }
}

impl From<&Piano> for SerializedPiano {
    fn from(value: &Piano) -> Self {
        SerializedPiano {
            strings: value.strings.iter().map(Into::into).collect(),
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn force_for_displacement(d: f32) -> f32 {
    // const MAX_NON_LINEAR_FORCE: f32 = 0.0001;
    // const CURVE_RES: usize = 1024;

    // static POINTS: OnceLock<Linear<Equidistant<f32>, [f32; CURVE_RES], Identity>> = OnceLock::new();

    // let curve = POINTS.get_or_init(|| {
    //     let mut points = [0.0; CURVE_RES];
    //     // let change_point = (MAX_SQRT_FORCE / MAX_EXPECTED_FORCE * CURVE_RES as f32) as usize;
    //     for i in 1..CURVE_RES {
    //         points[i] = ((i as f32 / CURVE_RES as f32 * MAX_NON_LINEAR_FORCE * 1000.0 + 0.1).powf(2.0) - 0.1.powf(2.0)) / 100.0;
    //     }

    //     Linear::builder()
    //         .elements(points)
    //         .equidistant::<f32>()
    //         .domain(0.0, MAX_NON_LINEAR_FORCE)
    //         .build()
    //         .unwrap()
    // });

    // curve.gen(d.abs()) * d.signum()

    d
}

impl PianoString {
    fn pluck(&mut self, vel: f32, pos: f32, width: f32) {
        self.state.damping = DAMPING_OPEN;
        self.state.on = true;
        self.state.is_active = true;
        self.state.is_ringing = true;
        self.state.ran_away = false;

        let vel = vel.clamp(0.0, 1.0);
        let place_to_pluck = (self.state.pos.len() as f32 * pos) as usize;
        let pluck_width = (self.state.pos.len() as f32 * width / 2.0) as usize;
        for i in (place_to_pluck - pluck_width)..=(place_to_pluck + pluck_width) {
            let distance_from_place_to_pluck = i as f32 - place_to_pluck as f32;
            let x = distance_from_place_to_pluck / pluck_width as f32;
            self.state.vel[i].y = 100_f32.powf(-x * x) * vel;
        }
    }

    fn get_tension_at(&self, i: usize) -> f32 {
        // FIXME: orthogonal vs compression wave
        let disp = self.state.pos[i].y;
        disp * self.conf.boost
    }

    fn listen_left(&self) -> f32 {
        self.get_tension_at(3)
    }

    fn listen_right(&self) -> f32 {
        self.get_tension_at(self.state.pos.len() - 4)
    }

    fn listen_unboosted(&self) -> f32 {
        self.state.pos[3].y
    }

    /// needs to be called before plucking
    fn resize(&mut self) {
        let new_size = self.conf.size as usize;
        assert!(new_size > 4);
        self.conf.size = new_size as u32;
        self.state.pos.truncate(new_size);
        for i in self.state.pos.len()..new_size {
            self.state.pos.push(Vec2 {
                x: i as f32,
                y: 0.0,
            })
        }
        self.state.vel.resize(new_size, Vec2::ZERO);

        // make sure the string is still attached horizontaly
        // and not flying away
        let last = self.state.pos.len() - 1;
        self.state.pos[last] = Vec2 {
            x: last as f32,
            y: 0.0,
        };
        self.state.pos[last - 1] = Vec2 {
            x: (last - 1) as f32,
            y: 0.0,
        };
        self.state.vel[last] = Vec2::ZERO;
        self.state.vel[last - 1] = Vec2::ZERO;
    }

    fn apply_pressure(&mut self, p: f32) {
        // FIXME: smooth front
        self.state.vel[8].y += p;
    }

    fn tick(&mut self, sustain: bool) {
        if !self.state.is_ringing {
            return;
        }
        if !sustain && !self.state.on {
            self.state.is_active = false;
        }
        for _ in 0..SUB_STEPS {
            self.advance(sustain);
        }
    }

    fn advance(&mut self, sustain: bool) {
        let mut is_now_active = false;
        let active_thresh = 0.0001;

        // apply velocity
        for i in 1..(self.state.pos.len() - 1) {
            self.state.pos[i] += self.state.vel[i] / self.conf.inertia;

            if self.state.pos[i].y.abs() > active_thresh {
                is_now_active = true;
            }

            // check for runaway
            if self.state.pos[i].y.abs() > 1000.0 {
                self.state.ran_away = true;
                println!("Ran away at {:?}", self.conf);

                self.reset();
                return;
            }
        }

        // apply tension and elasticity
        for i in 2..(self.state.pos.len() - 2) {
            let elastic_force = self.calc_bend_force(i);
            let tens_force = self.calc_tens_force(i);

            self.state.vel[i] += tens_force + elastic_force;
            self.state.vel[i] *= 1.0 - self.state.damping;

            if self.state.vel[i].length_squared() > active_thresh {
                is_now_active = true;
            }
        }

        self.state.is_ringing = is_now_active || self.state.on || sustain;
    }

    fn calc_tens_force(&self, i: usize) -> Vec2 {
        let mut res = Vec2::ZERO;
        for neighbour in [self.state.pos[i - 1], self.state.pos[i + 1]] {
            let displacement = 1.0 - self.state.pos[i].distance(neighbour) - self.conf.tension;

            let displacement_vec = self.state.pos[i] - neighbour;
            res += force_for_displacement(displacement * self.conf.tension) * displacement_vec
        }
        res
    }
    fn calc_bend_force(&self, i: usize) -> Vec2 {
        let mut res = Vec2::ZERO;
        for [prevprev, prev] in [[-2_isize, -1], [2, 1]] {
            /*
                        i-2        i-1
                         o----------o
                                ^    \
                                |     \ <- beam
                 affecting beam |      o
                                       i

            same on the other side
            */
            let affecting_beam = self.state.pos[i.saturating_add_signed(prev)]
                - self.state.pos[i.saturating_add_signed(prevprev)];
            let beam = self.state.pos[i] - self.state.pos[i.saturating_add_signed(prev)];

            // // this is correct but too expensive:
            // let mut bend = affecting_beam.angle_to(beam);
            // bend = (bend.abs() - bend_dead_zone).max(0.0) * bend.signum();
            // let correction_vec = beam.perp();

            let correction_vec = -beam.reject_from(affecting_beam);

            res += correction_vec
        }
        res * self.conf.elasticity
        
    }

    fn reset(&mut self) {
        self.state.pos.iter_mut().enumerate().for_each(|(i, p)| {
            // FIXME: dedup
            *p = Vec2 {
                x: i as f32,
                y: 0.0,
            }
        });
        self.state.vel.iter_mut().for_each(|p| *p = Vec2::ZERO);
    }

    fn ringing_clone(&self) -> Self {
        let mut clone = self.clone();
        clone.state.damping = DAMPING_OPEN;
        clone.reset();
        clone
    }
}

fn config_path() -> PathBuf {
    directories::BaseDirs::new()
        .unwrap()
        .config_dir()
        .join("riano.json")
}

fn main() {
    let (midi_sender, midi_receiver) = channel();

    let vis_buf = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(VIS_BUF_SIZE * 2)));
    let buf2 = Arc::clone(&vis_buf);

    // MIDI
    let midi_in = MidiInput::new("riano").unwrap();
    let in_ports = midi_in.ports();
    let device_to_search = std::env::args().nth(1);

    println!("Discovered MIDI devices:");
    for p in &in_ports {
        println!("{:?}", midi_in.port_name(p))
    }

    let midi_device = match device_to_search {
        None => {
            println!("Add part of the name as an argument");
            std::process::exit(1);
        }
        Some(arg_p) => in_ports
            .iter()
            .find(|p| midi_in.port_name(p).unwrap().contains(&arg_p))
            .unwrap_or_else(|| panic!("{arg_p:?} not found")),
    };

    println!("Connecting to {}", midi_in.port_name(midi_device).unwrap());

    let _conn = midi_in
        .connect(
            midi_device,
            "reading midi",
            move |_t, m, _| {
                let (m, _len) = MidiMsg::from_midi(m).unwrap();
                if let MidiMsg::ChannelVoice { msg, .. } = m {
                    midi_sender.send(msg).unwrap();
                }
            },
            (),
        )
        .unwrap();

    // AUDIO
    let host = cpal::default_host();
    let dev = host.default_output_device().unwrap();
    println!("Audio dev {:?}", dev.name());
    println!("Supported out: {:?}", dev.default_output_config().unwrap());
    println!("Supported in: {:?}", dev.default_input_config().unwrap());

    let mut playback_config: cpal::StreamConfig = dev.default_output_config().unwrap().into();

    // I do not understand how buffer size setting affects the actual buffer sizes.
    // The amount of samples is half for the playback (because of 2 channels) and a quarter for recording
    // Will configure something that works and deal with differently sized buffers in the code
    let rec_buffer_size = 256 * 2;
    let play_buffer_size = 256 * 2;
    let safety_margin = 4;

    playback_config.channels = 2;
    playback_config.buffer_size = BufferSize::Fixed(play_buffer_size as u32);

    let record_ringbuf = ringbuf::HeapRb::new(rec_buffer_size * safety_margin);
    let (mut record_sender, mut record_listener) = record_ringbuf.split();
    // a second listener-local quasi-ringbuffer
    let mut record_listener_receive_buffer =
        Vec::with_capacity(play_buffer_size / playback_config.channels as usize);
    let mut read_samples = 0;

    let sample_rate = playback_config.sample_rate.0;

    let mut record_config: cpal::StreamConfig = dev.default_input_config().unwrap().into();
    record_config.buffer_size = BufferSize::Fixed(rec_buffer_size as u32);
    record_config.channels = 1;
    record_config.sample_rate = playback_config.sample_rate;

    println!("Playback {playback_config:?}");
    println!("Recording {record_config:?}");

    let record_stream = dev
        .build_input_stream(
            &record_config,
            move |d: &[f32], _| {
                for s in d {
                    if record_sender.try_push(*s).is_err() {
                        // println!("Error sending rec {e}");
                    }
                }
            },
            |err| eprintln!("Error recording {err}"),
            None,
        )
        .unwrap();
    record_stream.play().unwrap();

    let piano_config: SerializedPiano = match fs_err::read_to_string(config_path()) {
        Ok(s) => serde_json::from_str(&s).unwrap(),
        Err(e) => {
            println!("Error loading piano file, using generated config: {e}\n");
            let mut strings = vec![];
            // bass strings
            for i in 0..50 {
                strings.push(StringConfig {
                    tension: 2.0,
                    inertia: 90.0,
                    elasticity: 1.0,
                    size: 128,
                    expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                    boost: 1.0,
                })
            }
            // middle
            for i in 50..70 {
                strings.push(StringConfig {
                    tension: 2.0,
                    inertia: 30.0,
                    elasticity: 0.4,
                    size: 64,
                    expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                    boost: 1.0,
                })
            }
            // treble
            for i in 70..90 {
                strings.push(StringConfig {
                    tension: 2.0,
                    inertia: [10.0, 9.0].lerp((i - 70) as f32 / 10.0),
                    elasticity: 0.1,
                    size: 32,
                    expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                    boost: 1.0,
                })
            }
            // super treble
            for i in 90..STRINGS_COUNT {
                strings.push(StringConfig {
                    tension: 0.1,
                    inertia: 10.0,
                    elasticity: 0.000001,
                    size: 20,
                    expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                    boost: 1.0,
                })
            }

            SerializedPiano { strings }
        }
    };

    let piano: Arc<RwLock<Piano>> = Arc::new(RwLock::new(Piano {
        strings: piano_config.strings.into_iter().map(Into::into).collect(),
        sustain: false,
        pluck_force: 1.0,
        vocoder_level: 0.0,
    }));

    let piano2 = Arc::clone(&piano);
    let string_calc_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let stream = dev
        .build_output_stream(
            &playback_config,
            move |data: &mut [f32], _| {
                while let Ok(msg) = midi_receiver.try_recv() {
                    {
                        let mut piano = piano.write().unwrap();
                        let pluck_force = piano.pluck_force;
                        println!("{msg:?}");

                        match msg {
                            ChannelVoiceMsg::NoteOn { note, velocity } => {
                                piano.strings[note as usize].pluck(
                                    velocity as f32 / 255.0 * pluck_force,
                                    0.4,
                                    0.3,
                                );
                            }
                            ChannelVoiceMsg::NoteOff { note, .. } => {
                                let sustain = piano.sustain;
                                let str = &mut piano.strings[note as usize];
                                str.state.on = false;
                                if !sustain {
                                    str.state.damping = DAMPING_MUTED;
                                }
                            }
                            ChannelVoiceMsg::ControlChange {
                                // control 64 is sustain
                                control: ControlChange::CC { control: 64, value },
                            } => {
                                if value > 64 {
                                    piano.sustain = true;
                                } else {
                                    piano.sustain = false;
                                    piano.strings.iter_mut().for_each(|s| {
                                        if !s.state.on {
                                            s.state.damping = DAMPING_MUTED;
                                        }
                                    });
                                }
                            }
                            _ => continue,
                        };
                    }
                }
                let buf_len = data.len();
                let samples_count = buf_len / playback_config.channels as usize;
                record_listener_receive_buffer.resize(samples_count, 0.0);

                if read_samples < samples_count {
                    let listened = record_listener.pop_slice(
                        &mut record_listener_receive_buffer[read_samples..samples_count],
                    );
                    read_samples += listened;
                }
                if read_samples < samples_count {
                    println!(
                        "Input is still filling the buffer: {}",
                        record_listener_receive_buffer.len()
                    );
                } else {
                    read_samples = 0;
                }

                {
                    let buffers = string_calc_pool.install(|| {
                        let mut piano = piano.write().unwrap();
                        let vocoder_lever = piano.vocoder_level;
                        let sustain = piano.sustain;
                        piano
                            .strings
                            .iter_mut()
                            .filter(|s| s.state.is_ringing)
                            .par_bridge()
                            .map(|s| {
                                (0..samples_count)
                                    .flat_map(|i| {
                                        if s.state.is_active {
                                            s.apply_pressure(
                                                record_listener_receive_buffer[i] * vocoder_lever,
                                            );
                                        }
                                        s.tick(sustain);
                                        [s.listen_left(), s.listen_right()]
                                    })
                                    .collect_vec()
                            })
                            .reduce(
                                || vec![0.0; buf_len],
                                |mut l, r| {
                                    l.iter_mut().zip(r).for_each(|(l, r)| *l += r);
                                    l
                                },
                            )
                    });

                    for (buffer_out, buffer_calc) in data.iter_mut().zip(buffers) {
                        *buffer_out = buffer_calc
                    }
                }

                let mut vis_buf = vis_buf.lock().unwrap();
                if vis_buf.len() < VIS_BUF_SIZE + VIS_BUF_SLACK {
                    vis_buf.extend(data.iter().copied());
                }
            },
            |err| eprintln!("{err:?}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();

    // VISUALS
    eframe::run_native(
        "Riano",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_inner_size([800.0, 500.0]),
            ..Default::default()
        },
        Box::new(move |_cc| {
            Box::new(App {
                vis_buf: buf2,
                locked: vec![],
                freq_detector: Arc::new(FreqDetector::new(4096 * 2, sample_rate as usize)),
                piano: piano2,
                sim_string: None,
                tune_channel: channel(),
                freq_meas_channel: channel(),
            })
        }),
    )
    .unwrap();
}

// FIXME: channel types are gnarly,
// replace them with a callback mechanism
#[allow(clippy::type_complexity)]
struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    piano: Arc<RwLock<Piano>>,
    freq_detector: Arc<FreqDetector>,
    sim_string: Option<(PianoString, bool)>,

    tune_channel: (
        Sender<(usize, f32, f32, f32)>,
        Receiver<(usize, f32, f32, f32)>,
    ),
    freq_meas_channel: (Sender<(usize, f32, f32)>, Receiver<(usize, f32, f32)>),
}

const BASIC_BOOST: f32 = 0.3;

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        if let Some((s, run)) = &mut self.sim_string {
            let mut open = true;
            Window::new("Stepping string simulation")
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("pluck").clicked() {
                            s.pluck(1.0, 0.3, 0.1);
                        }
                        if ui.button("step").clicked() || *run {
                            s.tick(false);
                        }
                        if ui.button("reset").clicked() {
                            s.reset();
                        }
                        ui.checkbox(run, "run");
                    });
                    Plot::new("sim string").show(ui, |ui| {
                        let pos_iter = s
                            .state
                            .pos
                            .iter()
                            .copied()
                            .map(|s| [s.x as f64, s.y as f64]);

                        ui.line(Line::new(pos_iter.clone().collect::<PlotPoints>()));
                        let vel = s
                            .state
                            .vel
                            .iter()
                            .copied()
                            .zip(&s.state.pos)
                            .map(|(v, p)| [(v.x + p.x) as f64, (v.y + p.y) as f64])
                            .collect::<PlotPoints>();
                        ui.arrows(
                            Arrows::new(pos_iter.clone().collect::<PlotPoints>(), vel)
                                .tip_length(10.0),
                        );
                    });
                });
            if !open {
                self.sim_string = None;
            }
        }

        let mut piano_clone = self.piano.read().unwrap().clone();

        SidePanel::left("strings settings").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let tune_bg = |piano: Arc<RwLock<Piano>>, i: usize, fd: Arc<FreqDetector>| {
                    let sender = self.tune_channel.0.clone();

                    rayon::spawn(move || {
                        let mut s = piano.read().unwrap().strings[i].ringing_clone();
                        let target_freq = s.conf.expected_frequency;
                        let (current_freq, level) = tune(&fd, &mut s, target_freq);
                        sender
                            .send((i, s.conf.tension, current_freq, level))
                            .unwrap();
                    })
                };

                let measure_bg = |piano: Arc<RwLock<Piano>>, i: usize, fd: Arc<FreqDetector>| {
                    let sender = self.freq_meas_channel.0.clone();
                    rayon::spawn(move || {
                        let mut s = piano.read().unwrap().strings[i].ringing_clone();
                        let (freq, spl) = measure(&mut s, &fd);
                        sender.send((i, freq, spl)).unwrap();
                    });
                };

                Grid::new("strings settings items")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("MIDI #");
                        ui.label("expected freq");
                        ui.label("current freq");
                        ui.group(|ui| {
                            if ui.button("tune all").clicked() {
                                for i in LOWEST_KEY..STRINGS_COUNT {
                                    tune_bg(Arc::clone(&self.piano), i, self.freq_detector.clone());
                                }
                            }
                            if ui.button("measure all").clicked() {
                                for i in 0..STRINGS_COUNT {
                                    measure_bg(self.piano.clone(), i, self.freq_detector.clone());
                                }
                            }
                        });
                        ui.end_row();

                        for (i, s) in piano_clone.strings.iter_mut().enumerate().skip(LOWEST_KEY) {
                            ui.label(RichText::new(format!("{i}")).background_color(
                                Color32::from_rgb(
                                    if s.state.ran_away { 255 } else { 0 },
                                    (s.state.pos.iter().map(|p| p.y).sum::<f32>().abs() * 255.0)
                                        as u8,
                                    0,
                                ),
                            ));
                            if ui
                                .label(format!("{:.1}", s.conf.expected_frequency))
                                .on_hover_text("tune")
                                .clicked()
                            {
                                tune_bg(self.piano.clone(), i, self.freq_detector.clone());
                            }
                            if ui
                                .label(format!("{:.1}", s.state.current_frequency))
                                .on_hover_text("measure")
                                .clicked()
                            {
                                measure_bg(self.piano.clone(), i, self.freq_detector.clone());
                            }
                            ui.vertical(|ui| {
                                ui.add(
                                    Slider::new(&mut s.conf.inertia, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("inertia"),
                                );
                                ui.add(
                                    Slider::new(&mut s.conf.tension, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("tension"),
                                );
                                ui.add(
                                    Slider::new(&mut s.conf.elasticity, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("elasticity"),
                                );

                                ui.add(Slider::new(&mut s.conf.size, 16..=256).text("size"));
                                ui.add(
                                    Slider::new(&mut s.conf.boost, 0.1..=10.0)
                                        .logarithmic(true)
                                        .text("boost"),
                                );
                                ui.separator();
                            });
                            if ui.button("Sim").clicked() {
                                let mut s = s.clone();
                                s.state.damping = DAMPING_OPEN;
                                self.sim_string = Some((s, false));
                            }
                            ui.end_row();
                        }
                    })
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            let mut reset_zoom = false;
            let mut buf = self.vis_buf.lock().unwrap();
            let to_remove = buf.len().saturating_sub(VIS_BUF_SIZE);
            for _ in 0..to_remove {
                buf.pop_front();
            }
            ui.horizontal(|ui| {
                if ui.button("lock").clicked() {
                    self.locked = buf.iter().copied().collect();
                }
                if ui.button("reset").clicked() {
                    self.locked.clear();
                }
                if ui.button("save").clicked() {
                    fs_err::write(
                        config_path(),
                        serde_json::to_string_pretty(&SerializedPiano::from(&piano_clone)).unwrap(),
                    )
                    .unwrap();
                }
                reset_zoom = ui.button("reset zoom").clicked();
            });
            if ui
                .button(
                    RichText::new("RESET STRINGS")
                        .color(Color32::RED)
                        .size(20.0),
                )
                .clicked()
            {
                self.piano
                    .write()
                    .unwrap()
                    .strings
                    .iter_mut()
                    .for_each(|s| s.reset());
            }
            ui.add(Slider::new(&mut piano_clone.pluck_force, 0.0..=10.0).text("pluck force"));
            ui.add(Slider::new(&mut piano_clone.vocoder_level, 0.0..=10.0).text("vocoder"));

            Plot::new("waveform")
                .auto_bounds(Vec2b { x: true, y: false })
                .allow_double_click_reset(false)
                .height(ui.available_height() / 2.0)
                .show(ui, |ui| {
                    if reset_zoom {
                        ui.set_plot_bounds(PlotBounds::from_min_max(
                            [0.0, -1.0],
                            [VIS_BUF_SECS as f64, 1.0],
                        ))
                    }
                    let step_by = 20;
                    let left: PlotPoints = buf
                        .iter()
                        .enumerate()
                        .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                        .step_by(step_by)
                        .collect();

                    let right: PlotPoints = buf
                        .iter()
                        .enumerate()
                        .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                        .skip(1)
                        .step_by(step_by)
                        .collect();

                    let locked: PlotPoints = self
                        .locked
                        .iter()
                        .enumerate()
                        .step_by(2)
                        .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                        .collect();

                    ui.line(Line::new(left).name("left"));
                    ui.line(Line::new(right).name("right"));
                    ui.line(Line::new(locked).name("locked"));
                });

            Plot::new("strings vis")
                .auto_bounds(Vec2b { x: true, y: false })
                .allow_double_click_reset(false)
                .show(ui, |ui| {
                    if reset_zoom {
                        ui.set_plot_bounds(PlotBounds::from_min_max([0.0, -1.0], [128.0, 1.0]))
                    }
                    for (i, string) in piano_clone
                        .strings
                        .iter()
                        .filter(|s| s.state.is_ringing)
                        .enumerate()
                    {
                        let line = Line::new(
                            string
                                .state
                                .pos
                                .iter()
                                .map(|v| [v.x as f64, v.y as f64 * string.conf.boost as f64])
                                .collect::<PlotPoints>(),
                        )
                        .name(&format!("string {i}"));
                        ui.line(line);
                    }
                });
        });

        let mut real_piano = self.piano.write().unwrap();
        real_piano.pluck_force = piano_clone.pluck_force;
        real_piano.vocoder_level = piano_clone.vocoder_level;

        for (real_s, new_s) in real_piano.strings.iter_mut().zip(&piano_clone.strings) {
            real_s.state.current_frequency = new_s.state.current_frequency;
            real_s.conf = new_s.conf.clone();
            real_s.resize();
        }

        while let Ok((i, freq, level)) = self.freq_meas_channel.1.try_recv() {
            let s = &mut real_piano.strings[i];
            s.state.current_frequency = freq;
            s.conf.boost = BASIC_BOOST / level.clamp(0.01, 100.0)
        }

        while let Ok((i, tens, freq, level)) = self.tune_channel.1.try_recv() {
            let s = &mut real_piano.strings[i];
            s.conf.tension = tens;
            s.state.current_frequency = freq;
            s.conf.boost = BASIC_BOOST / level.clamp(0.01, 100.0)
        }
    }
}

/// returns freq, signal level
fn measure(s: &mut PianoString, freq_detector: &FreqDetector) -> (f32, f32) {
    s.reset();
    s.pluck(0.3, 0.5, 0.6);
    // make sure the wave propagates through
    for _ in 0..20000 {
        s.tick(false);
    }

    let samples = (0..freq_detector.sample_count)
        .map(|_| {
            s.tick(false);
            s.listen_unboosted()
        })
        .collect_vec();

    let signal_level = samples
        .iter()
        .map(|s| s.abs())
        .max_by(|s1, s2| s1.total_cmp(s2))
        .unwrap();

    (freq_detector.detect(&samples), signal_level)
}

/// smarter frequency detection
#[derive(Clone)]
struct FreqDetector {
    fft: Arc<dyn Fft<f32>>,
    sample_count: usize,
    sample_rate: usize,
}

impl FreqDetector {
    fn new(sample_count: usize, sample_rate: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            fft: planner.plan_fft_forward(sample_count),

            sample_count,
            sample_rate,
        }
    }

    fn detect(&self, samples: &[f32]) -> f32 {
        let mut buf = samples
            .iter()
            .copied()
            .map(|s| Complex { re: s, im: 0.0 })
            .collect_vec();

        self.fft.process(&mut buf);

        let peak = buf
            .iter()
            .copied()
            .enumerate()
            .take(self.sample_count / 2)
            .max_by_key(|(_, s)| (s.abs() * 1000.0) as u32)
            .expect("to have at least 1 sample");
        if peak.1.abs() < 0.001 {
            return 0.0;
        }

        // use neighbors for anti-aliasing
        let mut neighbors = Vec::with_capacity(3);
        neighbors.push(peak);
        if peak.0 > 1 {
            neighbors.push((peak.0 - 1, buf[peak.0 - 1]));
        }
        if peak.0 < (self.sample_count / 2 - 1) {
            neighbors.push((peak.0 + 1, buf[peak.0 + 1]));
        }

        neighbors.sort_unstable_by(|c1, c2| c1.1.abs().total_cmp(&c2.1.abs()).reverse());
        // only take the two top values
        neighbors.truncate(2);
        // take weighted average of the two biggest
        let res = (self.fft_bucket_to_freq(neighbors[0].0) * neighbors[0].1.abs()
            + self.fft_bucket_to_freq(neighbors[1].0) * neighbors[1].1.abs())
            / (neighbors[0].1.abs() + neighbors[1].1.abs());
        if res.is_nan() {
            panic!("{neighbors:?}");
        } else {
            res
        }
    }

    fn fft_bucket_to_freq(&self, bucket: usize) -> f32 {
        bucket as f32 * self.sample_rate as f32 / self.sample_count as f32
    }
}

/// clone the string to avoid strumming the live strings
fn tune(freq_detector: &FreqDetector, s: &mut PianoString, target_freq: f32) -> (f32, f32) {
    let mut clamp = 0.2;
    let mut current_freq = 0.0;
    let mut level = 1.0;
    let mut prev_meas: Option<(f32, f32)> = None;
    let tries = 200;
    for _try in 0..tries {
        (current_freq, level) = measure(s, freq_detector);

        let error = target_freq - current_freq;
        if error.abs() < target_freq * 0.003 {
            println!("tuned");
            break;
        }
        if s.state.ran_away {
            println!("String ran away");
            level = 5.0;
            break;
        }
        let tension_adjust = match prev_meas {
            None => {
                // try something to measure the initial change in frequency
                error * 0.001
            }
            // try to predict the linear change
            Some((prev_tens, prev_freq)) => {
                let corr = calc_correction(
                    prev_freq,
                    current_freq,
                    target_freq,
                    prev_tens,
                    s.conf.tension,
                );

                let res = corr.clamp(-(s.conf.tension * clamp), s.conf.tension * clamp);
                clamp /= 1.05;
                res
            }
        };
        prev_meas = Some((s.conf.tension, current_freq));
        let overshot_protection = 0.9;
        s.conf.tension =
            (s.conf.tension + tension_adjust * overshot_protection).clamp(0.001, 100.0);
    }
    (current_freq, level)
}

fn calc_correction(
    prev_freq: f32,
    curr_freq: f32,
    target: f32,
    prev_tens: f32,
    curr_tens: f32,
) -> f32 {
    let mut freq_change = curr_freq - prev_freq;

    // div by zero protection
    if freq_change.abs() < 0.00001 {
        freq_change = 0.001 * freq_change.signum();
    }

    let tens_change = curr_tens - prev_tens;
    let freq_per_tens_change = tens_change / freq_change;
    let required_freq_change = target - curr_freq;

    let tens_correction = required_freq_change * freq_per_tens_change;

    println!("tens {prev_tens} to {curr_tens}, caused freq {prev_freq} -> {curr_freq} (want {target}), correcting {tens_correction} ");
    tens_correction
}

#[test]
fn adjust_smoke_test() {
    assert_eq!(calc_correction(10.0, 15.0, 13.0, 1.3, 1.5), -0.08000002);
    assert_eq!(calc_correction(15.0, 14.0, 10.0, 1.5, 1.45), -0.19999981);
    assert_eq!(calc_correction(150.0, 140.0, 160.0, 1.5, 1.49), 0.01999998);
}

#[test]
fn freq_detector_smoke_test() {
    use std::f32::consts::TAU;
    let samples = 4096 * 4;
    let freq_detector = FreqDetector::new(samples, 44100);

    for freq in [10, 20, 30, 100, 1000, 2000] {
        let sin_samples = (0..samples)
            .map(|i| {
                (i as f32 / 44100.0 * freq as f32 * TAU).sin()
                // noise
                + (i as f32 / 100.0).sin() * 0.1
                + (i as f32 / 120.0).sin() * 0.1
            })
            .collect_vec();

        let detected_freq = freq_detector.detect(&sin_samples);
        dbg!(detected_freq, freq);
        assert!(
            (detected_freq - freq as f32).abs() < 1.0,
            "detected {detected_freq} expected {freq}"
        );
    }
}
