use std::{
    array,
    cmp::Reverse,
    collections::VecDeque,
    f32::consts::{FRAC_PI_2, TAU},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    time::Instant,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use eframe::egui::{
    self, accesskit::Node, CentralPanel, Color32, Grid, RichText, SidePanel, Slider, Vec2b,
};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use itertools::Itertools;
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;
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
    expected_frequency: f32,
    current_frequency: f32,
    ran_away: bool,
    is_active: bool,
    tension: f32,
    inertia: f32,
    elasticity: f32,
    pos: Vec<f32>,
    vel: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SerializedPiano {
    strings: Vec<SerializedString>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializedString {
    size: u32,
    tension: f32,
    inertia: f32,
    elasticity: f32,
    expected_frequency: f32,
}

impl From<SerializedString> for PianoString {
    fn from(value: SerializedString) -> Self {
        let SerializedString {
            size,
            tension,
            inertia,
            elasticity,
            expected_frequency,
        } = value;
        PianoString {
            expected_frequency,
            current_frequency: 0.0,
            ran_away: false,
            is_active: false,
            tension,
            inertia,
            elasticity,
            pos: vec![0.0; size as usize],
            vel: vec![0.0; size as usize],
        }
    }
}

impl From<&PianoString> for SerializedString {
    fn from(value: &PianoString) -> Self {
        SerializedString {
            size: value.pos.len() as u32,
            tension: value.tension,
            inertia: value.inertia,
            elasticity: value.elasticity,
            expected_frequency: value.expected_frequency,
        }
    }
}

impl PianoString {
    fn pluck(&mut self, vel: f32, pos: f32, width: f32) {
        let vel = vel.clamp(0.0, 1.0);
        let place_to_pluck = (self.pos.len() as f32 * pos) as usize;
        let pluck_width = (self.pos.len() as f32 * width / 2.0) as usize;
        for i in (place_to_pluck - pluck_width)..=(place_to_pluck + pluck_width) {
            let distance_from_place_to_pluck = i as f32 - place_to_pluck as f32;
            self.vel[i] =
                (distance_from_place_to_pluck / pluck_width as f32 * FRAC_PI_2).cos() * vel;
        }
        self.is_active = true;
        self.ran_away = false;
    }

    fn listen(&self) -> f32 {
        self.pos[4]
    }

    /// needs to be called before plucking
    fn resize(&mut self, new_size: usize) {
        assert!(new_size > 4);
        self.pos.resize(new_size, 0.0);
        self.vel.resize(new_size, 0.0);

        // make sure the string is still attached horizontaly
        // and not flying away
        let last = self.pos.len() - 1;
        self.pos[last] = 0.0;
        self.pos[last - 1] = 0.0;
        self.vel[last] = 0.0;
        self.vel[last - 1] = 0.0;
    }

    fn tick(&mut self) {
        if !self.is_active {
            return;
        }

        let mut is_now_active = false;
        let active_thresh = 0.0001;

        let damping = 0.0001;

        // apply velocity
        for i in 1..(self.pos.len() - 1) {
            self.pos[i] += self.vel[i] / self.inertia;

            if self.pos[i].abs() > active_thresh {
                is_now_active = true;
            }
            // check for runaway
            if self.pos[i].abs() > 1000.0 {
                self.ran_away = true;
                println!(
                    "Runaway at tension {} inertia {} elasticity {}",
                    self.tension, self.inertia, self.elasticity
                );

                // println!("{:?}", self.pos);
                // println!("{:?}", self.vel);

                self.reset();
                return;
            }
        }

        // apply tension and elasticity
        for i in 2..(self.pos.len() - 2) {
            let straight_from_left = self.pos[i - 1] - self.pos[i - 2];
            let left_force_target = self.pos[i - 1] + straight_from_left * 2.0;

            let right_vec = self.pos[i + 1] - self.pos[i + 2];
            let right_vec_target = self.pos[i + 1] + right_vec * 2.0;

            let elastic_force_target = (left_force_target + right_vec_target) / 2.0;
            let tension_force_target = (self.pos[i - 1] + self.pos[i + 1]) / 2.0;

            let tension_force = tension_force_target - self.pos[i];
            let elastic_force = elastic_force_target - self.pos[i];

            self.vel[i] += tension_force * self.tension + elastic_force * self.elasticity;
            self.vel[i] *= 1.0 - damping;

            if self.vel[i].abs() > active_thresh {
                is_now_active = true;
            }
        }

        self.is_active = is_now_active;
    }

    fn reset(&mut self) {
        self.pos.iter_mut().for_each(|p| *p = 0.0);
        self.vel.iter_mut().for_each(|p| *p = 0.0);
    }
}

fn config_path() -> PathBuf {
    directories::BaseDirs::new()
        .unwrap()
        .config_dir()
        .join("riano.json")
}

fn main() {
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
                        SerializedString {
                            tension: [2.0, 0.3].lerp(perc),
                            inertia: [20.0, 1.0].lerp(perc),
                            elasticity: [0.3, 0.001].lerp(perc),
                            size: [128, 24].lerp(perc) as u32,
                            expected_frequency: 440.0 * 2_f32.powf((i as f32 - 69.0) / 12.0),
                        }
                    })
                    .collect(),
            }
        }
    };

    let strings: Arc<Mutex<Vec<PianoString>>> = Arc::new(Mutex::new(
        piano_config.strings.into_iter().map(Into::into).collect(),
    ));

    let strings_reader = Arc::clone(&strings);

    let mut start = Instant::now();

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                // care with deadlocks, do not lock anything, while holding this
                let mut strings = strings.lock().unwrap();
                while let Ok((string_n, vel)) = pluck_receiver.try_recv() {
                    let string = &mut strings[string_n as usize];
                    string.pluck(vel as f32 / 255.0, 0.3, 0.2);
                    println!(
                        "playing string {} tension {} inertia {} elasticity {}",
                        string_n, string.tension, string.inertia, string.elasticity
                    );
                }

                for c in data.chunks_exact_mut(2) {
                    strings.iter_mut().for_each(PianoString::tick);

                    // left
                    c[0] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.listen())
                        .sum::<f32>();
                    // right
                    c[1] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.listen())
                        .sum::<f32>();
                }
                drop(strings);

                let mut vis_buf = vis_buf.lock().unwrap();
                vis_buf.extend(data.iter().copied());
                if vis_buf.len() > VIS_BUF_SIZE * 2 {
                    panic!("visual buf not being consumed");
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
                freq_detector: FreqDetector::new(4096 * 8, sample_rate as usize),

                strings_editor: strings_reader,
            })
        }),
    )
    .unwrap();
}

struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    strings_editor: Arc<Mutex<Vec<PianoString>>>,
    freq_detector: FreqDetector,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        SidePanel::left("strings settings").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                Grid::new("strings settings items")
                    .striped(true)
                    .show(ui, |ui| {
                        let mut strings = self.strings_editor.lock().unwrap();
                        ui.label("MIDI #");

                        static STOP_TUNING: AtomicBool = AtomicBool::new(true);
                        if ui
                            .label("expected freq")
                            .on_hover_text("tune all")
                            .clicked()
                        {
                            let strings = Arc::clone(&self.strings_editor);
                            let fd2 = self.freq_detector.clone();

                            if STOP_TUNING.load(Ordering::Relaxed) {
                                STOP_TUNING.store(false, Ordering::Relaxed);
                                std::thread::spawn(move || {
                                    let strings_count = strings.lock().unwrap().len();
                                    for i in 0..strings_count {
                                        if STOP_TUNING.load(Ordering::Relaxed) {
                                            return;
                                        }
                                        let mut s = strings.lock().unwrap()[i].clone();
                                        let target_freq = s.expected_frequency;
                                        tune(&fd2, &mut s, target_freq);
                                        strings.lock().unwrap()[i].tension = s.tension;
                                    }
                                });
                            } else {
                                STOP_TUNING.store(true, Ordering::Relaxed);
                            }

                            // for s in strings.iter_mut() {
                            //     tune(&self.freq_detector, s, s.expected_frequency);
                            // }});
                        }
                        if ui
                            .label("current freq")
                            .on_hover_text("measure all")
                            .clicked()
                        {
                            for s in strings.iter_mut() {
                                s.current_frequency = measure(&mut s.clone(), &self.freq_detector);
                            }
                        }
                        ui.end_row();

                        for (i, s) in strings.iter_mut().enumerate() {
                            ui.label(RichText::new(format!("{i}")).background_color(
                                Color32::from_rgb(
                                    if s.ran_away { 255 } else { 0 },
                                    (s.pos.iter().sum::<f32>().abs() * 255.0) as u8,
                                    0,
                                ),
                            ));
                            if ui
                                .label(format!("{:.1}", s.expected_frequency))
                                .on_hover_text("tune")
                                .clicked()
                            {
                                let mut s2 = s.clone();
                                tune(&self.freq_detector, &mut s2, s.expected_frequency);
                                s.tension = s2.tension;
                            }
                            if ui
                                .label(format!("{:.1}", s.current_frequency))
                                .on_hover_text("detect")
                                .clicked()
                            {
                                let mut s_clone = s.clone();
                                s.current_frequency = measure(&mut s_clone, &self.freq_detector);
                            }
                            ui.vertical(|ui| {
                                ui.add(
                                    Slider::new(&mut s.inertia, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("inertia"),
                                );
                                ui.add(
                                    Slider::new(&mut s.tension, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("tension"),
                                );
                                ui.add(
                                    Slider::new(&mut s.elasticity, 0.0001..=100.0)
                                        .logarithmic(true)
                                        .text("elasticity"),
                                );
                                let mut size = s.pos.len();
                                ui.add(Slider::new(&mut size, 16..=256).text("size"));
                                if size != s.pos.len() {
                                    s.resize(size);
                                }
                                ui.separator();
                            });
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
                    let strings = self.strings_editor.lock().unwrap();
                    std::fs::write(
                        config_path(),
                        serde_json::to_string(&SerializedPiano {
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
                    for (i, string) in self.strings_editor.lock().unwrap().iter().enumerate() {
                        let line = Line::new(
                            string
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
            s.listen()
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
fn tune(freq_detector: &FreqDetector, s: &mut PianoString, target_freq: f32) {
    let mut prev_meas: Option<(f32, f32)> = None;
    for _try in 0..100 {
        let current_freq = measure(s, freq_detector);

        let error = target_freq - current_freq;
        if error.abs() < target_freq * 0.01 {
            println!("tuned");
            return;
        }
        if error.abs() > target_freq * 2.0 {
            println!(
                "String is too far from tune, cur: {} targ: {}",
                current_freq, target_freq
            );
            return;
        }
        let tension_adjust = match prev_meas {
            None => {
                // try something to measure the initial change in frequency
                error * 0.001
            }
            // try to predict the linear change
            Some((prev_tens, prev_freq)) => {
                let corr =
                    calc_correction(prev_freq, current_freq, target_freq, prev_tens, s.tension);

                corr.clamp(-(s.tension *0.1), s.tension * 0.2)
            }
        };
        prev_meas = Some((s.tension, current_freq));
        s.tension = (s.tension + tension_adjust).clamp(0.001, 100.0);
    }
    println!("Giving up");
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
    let samples = 4096 * 8;
    let freq_detector = FreqDetector::new(samples, 44100);

    for freq in [10, 100, 1000, 2000] {
        let sin_samples = (0..samples)
            .map(|i| {
                (i as f32 / 44100.0 * freq as f32 * TAU).sin()
                // noise
                + (i as f32 / 100.0).sin() * 0.1
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
