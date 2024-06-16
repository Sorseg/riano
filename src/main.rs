use std::{
    array,
    collections::VecDeque,
    f32::consts::FRAC_PI_2,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    time::Instant,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};

use eframe::egui::{self, CentralPanel, Slider, Vec2b};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use glam::Vec2;
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;

const VIS_BUF_SECS: f32 = 4.0;
const VIS_BUF_SIZE: usize = (44100.0 * VIS_BUF_SECS) as usize;

#[derive(Debug, PartialEq, Clone)]
struct Settings {
    boost: f32,
    inertia: [f32; 2],
    tension: [f32; 2],
    elasticity: [f32; 2],
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            boost: 1.0,
            inertia: [12.0, 1.0],
            tension: [1.0, 5.0],
            elasticity: [0.1, 0.001],
        }
    }
}

trait Lerp {
    fn lerp(&self, percent: f32) -> f32;
}

impl Lerp for [f32; 2] {
    fn lerp(&self, percent: f32) -> f32 {
        self[0] + (self[1] - self[0]) * percent
    }
}

struct PianoString<const N: usize> {
    is_active: bool,
    tension: f32,
    inertia: f32,
    elasticity: f32,
    // x,y
    pos: [Vec2; N],
    vel: [Vec2; N],
}

const STRING_POINTS: usize = 16;

impl<const N: usize> PianoString<N> {
    /// tension should be less than inertia doubled
    fn new(tension: f32, inertia: f32) -> Self {
        let inertia = inertia.clamp(0.1, 1000.0);
        let tension = tension.clamp(0.01, inertia * 1.99);

        Self {
            is_active: false,
            tension,
            inertia,
            elasticity: 0.001,
            pos: array::from_fn(|i| Vec2::new(i as f32, 0.0)),
            vel: array::from_fn(|_| Vec2::new(0.0, 0.0)),
        }
    }
    fn pluck(&mut self, vel: f32) {
        let vel = vel.clamp(0.0, 1.0);
        let place_to_pluck = 6.clamp(0, N);
        let pluck_width = 3.clamp(0, N);
        for i in (place_to_pluck - pluck_width)..=(place_to_pluck + pluck_width) {
            let distance_from_place_to_pluck = i as f32 - place_to_pluck as f32;
            self.vel[i].y =
                (distance_from_place_to_pluck / pluck_width as f32 * FRAC_PI_2).cos() * vel;
        }
        self.is_active = true;
    }
    fn listen(&self) -> f32 {
        self.pos[4].y
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

            if self.pos[i].y.abs() > active_thresh {
                is_now_active = true;
            }
            // check for runaway
            if self.pos[i].y.abs() > 1000.0 {
                println!(
                    "Runaway at tension {} inertia {}",
                    self.tension, self.inertia
                );

                println!("{:?}", self.pos);
                println!("{:?}", self.vel);

                self.vel.iter_mut().for_each(|p| *p = Vec2::ZERO);
                self.pos.iter_mut().for_each(|p| *p = Vec2::ZERO);
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

            if self.vel[i].y.abs() > active_thresh {
                is_now_active = true;
            }
        }

        self.is_active = is_now_active;
    }
}

fn main() {
    let (pluck_sender, pluck_receiver) = channel();

    let vis_buf = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buf2 = Arc::clone(&vis_buf);

    let strings_snapshots_writer: Arc<Mutex<Vec<Vec<Vec2>>>> = Arc::new(Mutex::new(vec![]));
    let strings_snapshots_reader = Arc::clone(&strings_snapshots_writer);
    let strings_snapshot_refresh_ms = 10;

    let (settings_sender, settings_reader) = channel::<Settings>();

    // MIDI
    let midi_in = MidiInput::new("riano").unwrap();
    let in_ports = midi_in.ports();
    let casio = in_ports
        .iter()
        .find(|p| midi_in.port_name(p).unwrap().contains("CASIO"))
        .expect("casio not connected");

    println!("Connecting to {}", midi_in.port_name(casio).unwrap());

    let _conn = midi_in
        .connect(
            casio,
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
    println!("{config:?}");

    let mut strings: [PianoString<STRING_POINTS>; 128] =
        array::from_fn(|_| PianoString::new(1.0, 10.0));

    let mut start = Instant::now();

    let mut boost = 1.0;

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                while let Ok((string_n, vel)) = pluck_receiver.try_recv() {
                    let string = &mut strings[string_n as usize];
                    string.pluck(vel as f32 / 255.0);
                    println!(
                        "playing string {} tension {} inertia {} elasticity {}",
                        string_n, string.tension, string.inertia, string.elasticity
                    );
                }
                if let Ok(setting) = settings_reader.try_recv() {
                    boost = setting.boost;

                    let denominator = (strings.len() - 1) as f32;
                    for (i, string) in strings.iter_mut().enumerate() {
                        let percent = i as f32 / denominator;
                        string.inertia = setting.inertia.lerp(percent);
                        string.tension = setting.tension.lerp(percent);
                        string.elasticity = setting.elasticity.lerp(percent);
                    }
                }

                for c in data.chunks_exact_mut(2) {
                    strings.iter_mut().for_each(PianoString::tick);

                    if start.elapsed().as_millis() >= strings_snapshot_refresh_ms {
                        start = Instant::now();
                        // takes ~5 us for 10 strings
                        let snap = strings
                            .iter()
                            .filter(|s| s.is_active)
                            .map(|s| s.pos.to_vec())
                            .collect();
                        *strings_snapshots_writer.lock().unwrap() = snap;
                    }
                    // left
                    c[0] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.listen())
                        .sum::<f32>()
                        * boost;
                    // right
                    c[1] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.listen())
                        .sum::<f32>()
                        * boost;
                }

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
        Box::new(|_cc| {
            settings_sender.send(Settings::default()).unwrap();

            Box::new(App {
                vis_buf: buf2,
                locked: vec![],
                settings_sender,
                settings: Settings::default(),
                strings_snapshots_reader,
            })
        }),
    )
    .unwrap();
}

struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    settings_sender: Sender<Settings>,
    settings: Settings,
    strings_snapshots_reader: Arc<Mutex<Vec<Vec<Vec2>>>>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        let mut buf = self.vis_buf.lock().unwrap();
        let to_remove = buf.len().saturating_sub(VIS_BUF_SIZE);
        for _ in 0..to_remove {
            buf.pop_front();
        }

        CentralPanel::default().show(ctx, |ui| {
            let mut reset_zoom = false;

            ui.horizontal(|ui| {
                if ui.button("lock").clicked() {
                    self.locked = buf.iter().copied().collect();
                }
                if ui.button("reset").clicked() {
                    self.locked.clear();
                }
                reset_zoom = ui.button("reset zoom").clicked();
            });

            let prev_setting = self.settings.clone();
            let width = ui.available_width() * 0.8;
            let size = egui::Vec2::new(width, 5.0);
            ui.style_mut().spacing.slider_width = 300.0;
            ui.add_sized(
                size,
                Slider::new(&mut self.settings.boost, 0.1..=5.0).text("boost"),
            );
            ui.group(|ui| {
                for (val, name) in [
                    (&mut self.settings.inertia, "inertia"),
                    (&mut self.settings.tension, "tension"),
                    (&mut self.settings.elasticity, "elasticity"),
                ] {
                    ui.add_sized(
                        size,
                        Slider::new(&mut val[0], 0.001..=100.0)
                            .logarithmic(true)
                            .text(format!("lowest {name}")),
                    );
                    ui.add_sized(
                        size,
                        Slider::new(&mut val[1], 0.001..=100.0)
                            .logarithmic(true)
                            .text(format!("highest {name}")),
                    );
                    ui.separator();
                }
            });

            if prev_setting != self.settings {
                self.settings_sender.send(self.settings.clone()).unwrap();
            }

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

            Plot::new("strings")
                .auto_bounds(Vec2b { x: true, y: false })
                .allow_double_click_reset(false)
                .show(ui, |ui| {
                    if reset_zoom {
                        ui.set_plot_bounds(PlotBounds::from_min_max(
                            [0.0, -3.0],
                            [STRING_POINTS as f64, 3.0],
                        ))
                    }
                    for (i, string) in self
                        .strings_snapshots_reader
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let line = Line::new(
                            string
                                .iter()
                                .map(|p| [p.x as f64, p.y as f64 * self.settings.boost as f64])
                                .collect::<PlotPoints>(),
                        )
                        .name(&format!("string {i}"));
                        ui.line(line);
                    }
                });
        });
    }
}
