use std::{
    collections::VecDeque,
    f32::consts::FRAC_PI_2,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::channel,
        Arc, Mutex, RwLock,
    },
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use eframe::egui::{self, CentralPanel, Color32, Grid, RichText, SidePanel, Slider, Vec2b, Window};
use egui_plot::{Arrows, Line, Plot, PlotBounds, PlotPoints};
use itertools::Itertools;
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    Fft, FftPlanner,
};
use serde::{Deserialize, Serialize};

const VIS_BUF_SECS: f32 = 4.0;
const VIS_BUF_SIZE: usize = (44100.0 * VIS_BUF_SECS) as usize;

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
struct PianoString {
    conf: StringConfig,
    state: StringState,
}

#[derive(Clone)]
struct StringState {
    current_frequency: f32,
    pos: Vec<f32>,
    vel: Vec<f32>,
    ran_away: bool,
    is_active: bool,
}

#[derive(Serialize, Deserialize)]
struct SerializedPiano {
    strings: Vec<StringConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StringConfig {
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
                current_frequency: 0.0,
                ran_away: false,
                is_active: false,
                pos: vec![0.0; value.size as usize],
                vel: vec![0.0; value.size as usize],
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

impl PianoString {
    fn pluck(&mut self, vel: f32, pos: f32, width: f32) {
        let vel = vel.clamp(0.0, 1.0);
        let place_to_pluck = (self.state.pos.len() as f32 * pos) as usize;
        let pluck_width = (self.state.pos.len() as f32 * width / 2.0) as usize;
        for i in (place_to_pluck - pluck_width)..=(place_to_pluck + pluck_width) {
            let distance_from_place_to_pluck = i as f32 - place_to_pluck as f32;
            self.state.vel[i] =
                (distance_from_place_to_pluck / pluck_width as f32 * FRAC_PI_2).cos() * vel;
        }
        self.state.is_active = true;
        self.state.ran_away = false;
    }

    fn listen_left(&self) -> f32 {
        self.state.pos[(self.state.pos.len() as f32 * 0.2) as usize]
    }
    fn listen_right(&self) -> f32 {
        self.state.pos[(self.state.pos.len() as f32 * 0.25) as usize]
    }

    /// needs to be called before plucking
    fn resize(&mut self, new_size: usize) {
        assert!(new_size > 4);
        self.conf.size = new_size as u32;
        self.state.pos.resize(new_size, 0.0);
        self.state.vel.resize(new_size, 0.0);

        // make sure the string is still attached horizontaly
        // and not flying away
        let last = self.state.pos.len() - 1;
        self.state.pos[last] = 0.0;
        self.state.pos[last - 1] = 0.0;
        self.state.vel[last] = 0.0;
        self.state.vel[last - 1] = 0.0;
    }

    fn tick(&mut self) {
        let st = &mut self.state;

        if !st.is_active {
            return;
        }

        let mut is_now_active = false;
        let active_thresh = 0.0001;

        let damping = 0.0001;

        // apply velocity
        for i in 1..(st.pos.len() - 1) {
            st.pos[i] += st.vel[i] / self.conf.inertia;

            if st.pos[i].abs() > active_thresh {
                is_now_active = true;
            }
            // check for runaway
            if st.pos[i].abs() > 1000.0 {
                st.ran_away = true;
                println!("Runaway at {:?}", self.conf);

                self.reset();
                return;
            }
        }

        // apply tension and elasticity
        for i in 2..(st.pos.len() - 2) {
            let straight_from_left = st.pos[i - 1] - st.pos[i - 2];
            let left_force_target = st.pos[i - 1] + straight_from_left * 2.0;

            let right_vec = st.pos[i + 1] - st.pos[i + 2];
            let right_vec_target = st.pos[i + 1] + right_vec * 2.0;

            let elastic_force_target = (left_force_target + right_vec_target) / 2.0;
            let tension_force_target = (st.pos[i - 1] + st.pos[i + 1]) / 2.0;

            let tension_force = tension_force_target - st.pos[i];
            let elastic_force = elastic_force_target - st.pos[i];

            st.vel[i] += tension_force * self.conf.tension + elastic_force * self.conf.elasticity;
            st.vel[i] *= 1.0 - damping;

            if st.vel[i].abs() > active_thresh {
                is_now_active = true;
            }
        }

        st.is_active = is_now_active;
    }

    fn reset(&mut self) {
        self.state.pos.iter_mut().for_each(|p| *p = 0.0);
        self.state.vel.iter_mut().for_each(|p| *p = 0.0);
    }
}

fn config_path() -> PathBuf {
    directories::BaseDirs::new()
        .unwrap()
        .config_dir()
        .join("riano.json")
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build_global()
        .unwrap();
    let (pluck_sender, pluck_receiver) = channel();

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
                if let MidiMsg::ChannelVoice {
                    msg: ChannelVoiceMsg::NoteOn { velocity, note },
                    ..
                } = m
                {
                    pluck_sender.send((note, velocity)).unwrap();
                }

                if let MidiMsg::ChannelVoice {
                    msg: ChannelVoiceMsg::NoteOff { .. },
                    ..
                } = m
                {}
            },
            (),
        )
        .unwrap();

    // AUDIO
    let host = cpal::default_host();
    let dev = host.default_output_device().unwrap();
    let mut config: cpal::StreamConfig = dev.default_output_config().unwrap().into();
    config.buffer_size = BufferSize::Fixed(256);
    assert_eq!(config.channels, 2);
    let sample_rate = config.sample_rate.0;
    println!("{config:?}");

    let piano_config: SerializedPiano = match std::fs::read_to_string(config_path()) {
        Ok(s) => serde_json::from_str(&s).unwrap(),
        Err(e) => {
            println!("Error loading config file: {e}");
            SerializedPiano {
                strings: (0..128)
                    .map(|i| {
                        let perc = i as f32 / 128.0;
                        StringConfig {
                            tension: [2.0, 0.3].lerp(perc),
                            inertia: [10.0, 1.0].lerp(perc),
                            elasticity: [0.1, 0.001].lerp(perc),
                            size: [64, 24].lerp(perc) as u32,
                            expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                        }
                    })
                    .collect(),
            }
        }
    };

    let strings: Arc<RwLock<Vec<PianoString>>> = Arc::new(RwLock::new(
        piano_config.strings.into_iter().map(Into::into).collect(),
    ));

    let strings_reader = Arc::clone(&strings);
    let string_calc_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                while let Ok((string_n, vel)) = pluck_receiver.try_recv() {
                    {
                        strings.write().unwrap()[string_n as usize].pluck(
                            vel as f32 / 255.0,
                            0.3,
                            0.2,
                        );
                        println!("playing string {}", string_n);
                    }
                }

                let buf_len = data.len();
                {
                    let buffers = string_calc_pool.install(|| {
                        strings
                            .write()
                            .unwrap()
                            .par_iter_mut()
                            .filter(|s| s.state.is_active)
                            .map(|s| {
                                (0..(buf_len / 2))
                                    .flat_map(|_| {
                                        s.tick();
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

                // let mut vis_buf = vis_buf.lock().unwrap();
                // vis_buf.extend(data.iter().copied());
                // if vis_buf.len() > VIS_BUF_SIZE * 2 {
                //     panic!("visual buf not being consumed");
                // }
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
                freq_detector: FreqDetector::new(4096 * 8, sample_rate as usize),
                strings_editor: strings_reader,
                sim_string: None,
            })
        }),
    )
    .unwrap();
}

struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    strings_editor: Arc<RwLock<Vec<PianoString>>>,
    freq_detector: FreqDetector,
    sim_string: Option<PianoString>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        if let Some(s) = &mut self.sim_string {
            let mut open = true;
            Window::new("Stepping string simulation")
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("pluck").clicked() {
                            s.pluck(1.0, 0.3, 0.1);
                        }
                        if ui.button("step").clicked() {
                            s.tick();
                        }
                        if ui.button("reset").clicked() {
                            s.reset();
                        }
                    });
                    Plot::new("sim string").show(ui, |ui| {
                        let pos_iter = s
                            .state
                            .pos
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(i, s)| [i as f64, s as f64]);

                        ui.line(Line::new(pos_iter.clone().collect::<PlotPoints>()));
                        let vel = s
                            .state
                            .vel
                            .iter()
                            .copied()
                            .enumerate()
                            .zip(&s.state.pos)
                            .map(|((i, v), p)| [i as f64, (v + *p) as f64])
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

        SidePanel::left("strings settings").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                Grid::new("strings settings items")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("MIDI #");

                        static STOP_TUNING: AtomicBool = AtomicBool::new(true);
                        if ui
                            .label("expected freq")
                            .on_hover_text("tune all")
                            .clicked()
                        {
                            if STOP_TUNING.load(Ordering::Relaxed) {
                                STOP_TUNING.store(false, Ordering::Relaxed);
                                // let strings_n = self.strings_editor.read().unwrap().len();
                                let strings_n = 128;

                                for i in 0..strings_n {
                                    let strings = Arc::clone(&self.strings_editor);
                                    let fd = Arc::new(self.freq_detector.clone());
                                    rayon::spawn(move || {
                                        if STOP_TUNING.load(Ordering::Relaxed) {
                                            return;
                                        }
                                        let mut s = strings.read().unwrap()[i].clone();
                                        let target_freq = s.conf.expected_frequency;
                                        let current_freq = tune(&fd, &mut s, target_freq);
                                        let actual_s = &mut strings.write().unwrap()[i];
                                        actual_s.conf.tension = s.conf.tension;
                                        actual_s.state.current_frequency = current_freq;
                                    });
                                }
                            } else {
                                STOP_TUNING.store(true, Ordering::Relaxed);
                            }
                        }
                        if ui
                            .label("current freq")
                            .on_hover_text("measure all")
                            .clicked()
                        {
                            let s_num = 128;
                            let fd = Arc::new(self.freq_detector.clone());
                            for i in 0..s_num {
                                let strings = Arc::clone(&self.strings_editor);
                                let fd = Arc::clone(&fd);
                                rayon::spawn(move || {
                                    let mut s = { strings.read().unwrap()[i].clone() };

                                    let res = measure(&mut s, &fd);
                                    strings.write().unwrap()[i].state.current_frequency = res;
                                });
                            }
                        }
                        ui.end_row();
                        // FIXME: write conf after the render
                        for (i, s) in self.strings_editor.write().unwrap().iter_mut().enumerate() {
                            ui.label(RichText::new(format!("{i}")).background_color(
                                Color32::from_rgb(
                                    if s.state.ran_away { 255 } else { 0 },
                                    (s.state.pos.iter().sum::<f32>().abs() * 255.0) as u8,
                                    0,
                                ),
                            ));
                            if ui
                                .label(format!("{:.1}", s.conf.expected_frequency))
                                .on_hover_text("tune")
                                .clicked()
                            {
                                let mut s2 = s.clone();
                                s.state.current_frequency =
                                    tune(&self.freq_detector, &mut s2, s.conf.expected_frequency);
                                s.conf.tension = s2.conf.tension;
                            }
                            if ui
                                .label(format!("{:.1}", s.state.current_frequency))
                                .on_hover_text("detect")
                                .clicked()
                            {
                                let mut s_clone = s.clone();
                                s.state.current_frequency =
                                    measure(&mut s_clone, &self.freq_detector);
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
                                let mut size = s.conf.size;
                                ui.add(Slider::new(&mut size, 16..=256).text("size"));
                                if size != s.conf.size {
                                    s.resize(size as usize);
                                }
                                ui.separator();
                            });
                            if ui.button("Sim").clicked() {
                                self.sim_string = Some(s.clone());
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
                    let strings = self.strings_editor.read().unwrap();
                    std::fs::write(
                        config_path(),
                        serde_json::to_string_pretty(&SerializedPiano {
                            strings: strings.iter().map(Into::into).collect(),
                        })
                        .unwrap(),
                    )
                    .unwrap();
                }
                reset_zoom = ui.button("reset zoom").clicked();
            });

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
                    for (i, string) in self
                        .strings_editor
                        .read()
                        .unwrap()
                        .iter()
                        .filter(|s| s.state.is_active)
                        .enumerate()
                    {
                        let line = Line::new(
                            string
                                .state
                                .pos
                                .iter()
                                .enumerate()
                                .map(|(i, v)| [i as f64, *v as f64])
                                .collect::<PlotPoints>(),
                        )
                        .name(&format!("string {i}"));
                        ui.line(line);
                    }
                });
        });
    }
}

fn measure(s: &mut PianoString, freq_detector: &FreqDetector) -> f32 {
    s.reset();
    s.pluck(0.2, 0.5, 0.5);
    // make sure the wave propagates through
    for _ in 0..40000 {
        s.tick();
    }
    let samples = (0..freq_detector.sample_count)
        .map(|_| {
            s.tick();
            s.listen_left()
        })
        .collect_vec();

    freq_detector.detect(&samples)
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
fn tune(freq_detector: &FreqDetector, s: &mut PianoString, target_freq: f32) -> f32 {
    let mut current_freq = 0.0;
    let mut prev_meas: Option<(f32, f32)> = None;
    for _try in 0..100 {
        current_freq = measure(s, freq_detector);

        let error = target_freq - current_freq;
        if error.abs() < target_freq * 0.005 {
            println!("tuned");
            break;
        }
        if s.state.ran_away {
            println!("String ran away");
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

                corr.clamp(-(s.conf.tension * 0.1), s.conf.tension * 0.2)
            }
        };
        prev_meas = Some((s.conf.tension, current_freq));
        s.conf.tension = (s.conf.tension + tension_adjust).clamp(0.001, 100.0);
    }
    current_freq
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
    // to avoid overshoot
    tens_correction * 0.5
}

#[test]
fn adjust_smoke_test() {
    assert_eq!(calc_correction(10.0, 15.0, 13.0, 1.3, 1.5), -0.04000001);
    assert_eq!(calc_correction(15.0, 14.0, 10.0, 1.5, 1.45), -0.099999905);
    assert_eq!(calc_correction(150.0, 140.0, 100.0, 1.5, 1.49), -0.01999998);
}

#[test]
fn freq_detector_smoke_test() {
    use std::f32::consts::TAU;
    let samples = 4096 * 8;
    let freq_detector = FreqDetector::new(samples, 44100);

    for freq in [10, 100, 1000, 2000] {
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
