use std::{
    array,
    collections::VecDeque,
    f32::consts::TAU,
    sync::{
        atomic::{AtomicI32, Ordering},
        mpsc::channel,
        Arc, Mutex,
    },
    time::Instant,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use derivative::Derivative;
use eframe::egui::{CentralPanel, Slider, Ui, Vec2b};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use glam::Vec2;
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;

#[derive(Clone, Derivative)]
#[derivative(Debug)]
struct Note {
    note_n: u8,
    // index in the samples array
    phase: f32,
    freq: f32,
    vel: f32,
    on: bool,
    /// 0.0 - left, 1.0 - right
    pan: f32,

    #[derivative(Debug = "ignore")]
    samples_at_20_hz: Arc<[f32; 44100 / 20]>,
}

const VIS_BUF_SECS: f32 = 4.0;
const VIS_BUF_SIZE: usize = (44100.0 * VIS_BUF_SECS) as usize;

#[derive(Debug)]
struct Settings {
    fat_phase_offset: AtomicI32,
    fat_detune: AtomicI32,
    fat_n: AtomicI32,
    fat_pan: AtomicI32,
    boost: AtomicI32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            fat_phase_offset: Default::default(),
            fat_detune: Default::default(),
            fat_n: 1.into(),
            fat_pan: 6000.into(),
            boost: Default::default(),
        }
    }
}

struct PianoString<const N: usize> {
    is_active: bool,
    tension: f32,
    inertia: f32,
    // x,y
    pos: [Vec2; N],
    vel: [Vec2; N],
}
impl<const N: usize> PianoString<N> {
    fn new(tension: f32, inertia: f32) -> Self {
        Self {
            is_active: false,
            tension: tension.clamp(0.1, 10.0),
            inertia: inertia.clamp(1.0, 1000.0),
            pos: array::from_fn(|i| Vec2::new(i as f32, 0.0)),
            vel: array::from_fn(|_| Vec2::new(0.0, 0.0)),
        }
    }
    fn pluck(&mut self, vel: f32) {
        let vel = vel.clamp(0.0, 1.0);
        for i in 0..N / 4 {
            self.vel[i].y = (i as f32 / N as f32 / 4.0) * vel;
        }
        for i in N / 4..N {
            self.vel[i].y = (N as f32 - i as f32) / N as f32 * vel;
        }
        self.is_active = true;
    }
    fn tick(&mut self) {
        if !self.is_active {
            return;
        }

        let mut is_active = false;
        let active_thresh = 0.0001;
        let damping = 0.0001;

        // apply velocity
        for i in 1..(self.pos.len() - 1) {
            self.pos[i] += self.vel[i] / self.inertia;

            if self.pos[i].y.abs() > active_thresh {
                is_active = true;
            }
        }

        // apply elasticity
        for i in 1..(self.pos.len() - 1) {
            let force = (self.pos[i - 1] + self.pos[i + 1]) / 2.0 - self.pos[i];
            self.vel[i] += force * self.tension;
            self.vel[i] *= 1.0 - damping;

            if self.vel[i].y.abs() > 50.0 {
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

            if self.vel[i].y.abs() > active_thresh {
                is_active = true;
            }
        }

        self.is_active = is_active;
    }
}

fn main() {
    let (impulse_sender, impulse_receiver) = channel();

    let vis_buf = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buf2 = Arc::clone(&vis_buf);

    let settings_writer = Arc::new(Settings::default());
    let settings_reader = Arc::clone(&settings_writer);

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
                    impulse_sender.send((note, velocity)).unwrap();
                }

                if let MidiMsg::ChannelVoice {
                    msg: ChannelVoiceMsg::NoteOff { note, .. },
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

    let mut strings: [PianoString<64>; 128] =
        array::from_fn(|i| PianoString::new(i as f32 / 10.0 - 5.0, 30.0 - i as f32 / 2.0));
    let mut start = Instant::now();

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                while let Ok((string_n, vel)) = impulse_receiver.try_recv() {
                    let string = &mut strings[string_n as usize];
                    string.pluck(vel as f32 / 255.0);
                    println!(
                        "playing  string {} tension {} inertia {}",
                        string_n, string.tension, string.inertia
                    );
                }
                let amplification = 0.2;

                for c in data.chunks_exact_mut(2) {
                    for s in &mut strings {
                        s.tick();
                    }
                    if start.elapsed().as_millis() > 100 {
                        // println!("{:?}", strings[50].pos);
                        start = Instant::now();
                    }
                    // left
                    c[0] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.pos[20].y)
                        .sum::<f32>()
                        * amplification;
                    // right
                    c[1] = strings
                        .iter()
                        .filter(|s| s.is_active)
                        .map(|s| s.pos[22].y)
                        .sum::<f32>()
                        * amplification;
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
        eframe::NativeOptions::default(),
        Box::new(|_cc| {
            Box::new(App {
                vis_buf: buf2,
                locked: vec![],
                settings: settings_writer,
            })
        }),
    )
    .unwrap();
}

struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    settings: Arc<Settings>,
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

            let slider = |ui: &mut Ui, atomic: &AtomicI32, range, text: &str| {
                let mut val = atomic.load(Ordering::Relaxed);
                let prev_value = val;
                ui.add(Slider::new(&mut val, range).text(text));
                if val != prev_value {
                    atomic.store(val, Ordering::Relaxed);
                }
            };
            // slider(
            //     ui,
            //     &self.settings.fat_phase_offset,
            //     -30000..=30000,
            //     "phase offset",
            // );
            // slider(ui, &self.settings.fat_detune, -10000..=10000, "detune");
            // slider(ui, &self.settings.fat_n, 1..=100, "chorun N");
            // slider(ui, &self.settings.fat_pan, -10000..=10000, "chorun pan");
            // slider(ui, &self.settings.boost, 0..=100000, "boost");

            Plot::new("waveform")
                .auto_bounds(Vec2b { x: true, y: false })
                .allow_double_click_reset(false)
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
        });
    }
}
