use std::{
    array,
    collections::VecDeque,
    f32::consts::TAU,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc, Mutex,
    },
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use derivative::Derivative;
use eframe::egui::{CentralPanel, Slider, Ui, Vec2b};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
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
    asdr: Asdr,
    on: bool,
    /// 0.0 - left, 1.0 - right
    pan: f32,

    #[derivative(Debug = "ignore")]
    samples_at_20_hz: Arc<[f32; 44100 / 20]>,
}

#[derive(Debug, Clone)]
struct Asdr {
    attack_level: f32,
    release_level: f32,
    attack: f32,
    release: f32,
}

impl Asdr {
    fn attack(&mut self, t: f32) -> f32 {
        if self.attack_level < 1.0 {
            self.attack_level += t / self.attack;
        }

        self.attack_level.clamp(0.0, 1.0)
    }

    fn release(&mut self, t: f32) -> f32 {
        // TODO: separate mutation and reading
        self.release_level *= 0.1_f32.powf(t / self.release);
        self.release_level
    }

    fn new(attack: f32, release: f32) -> Self {
        Self {
            attack,
            release,
            attack_level: 0.0,
            release_level: 1.0,
        }
    }
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

fn main() {
    // FIXME: opt
    let notes = Arc::new(Mutex::new(Vec::<Note>::new()));
    let notes2 = Arc::clone(&notes);
    let notes3 = Arc::clone(&notes);

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
    let mut note_counter: u64 = 1;
    let sin_pow_at_20_hz = Arc::new(array::from_fn(|i| {
        (i as f32 * 20.0 / 44100.0 * TAU).sin().powf(3.0)
    }));
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
                    // PRNG
                    let detune = (note as f32 * 1.523644) % 1.0;
                    let note = Note {
                        note_n: note,
                        on: true,
                        freq: 440.0 * 2_f32.powf((note as f32 + detune * 0.1 - 69.0) / 12.0),
                        vel: velocity as f32 / 255.0,
                        phase: 0.0,
                        asdr: Asdr::new(0.01, 1.0),
                        pan: 0.5,
                        samples_at_20_hz: Arc::clone(&sin_pow_at_20_hz),
                    };

                    let fat_n = settings_reader.fat_n.load(Ordering::Relaxed);

                    for fat in 0..fat_n {
                        // 0.0 .. 1.0
                        let fat_id = if fat_n == 1 {
                            0.5
                        } else {
                            fat as f32 / (fat_n - 1) as f32
                        };

                        let mut note = note.clone();
                        note.freq *= 1.0
                            + fat as f32
                                * (settings_reader.fat_detune.load(Ordering::Relaxed) as f32
                                    / 10000.0);
                        note.pan = 0.5
                            + (fat_id - 0.5)
                                * settings_reader.fat_pan.load(Ordering::Relaxed) as f32
                                / 10000.0;
                        note.vel *= (1.0
                            + settings_reader.boost.load(Ordering::Relaxed) as f32 / 10000.0)
                            / fat_n as f32;
                        note.phase = fat_id
                            * (settings_reader.fat_phase_offset.load(Ordering::Relaxed) as f32
                                / 10000.0)
                            * TAU;

                        note_counter += 1;
                        println!("{note_counter} {note:?}");

                        notes.lock().unwrap().push(note)
                    }
                }
                if let MidiMsg::ChannelVoice {
                    msg: ChannelVoiceMsg::NoteOff { note, .. },
                    ..
                } = m
                {
                    notes.lock().unwrap().iter_mut().for_each(|n| {
                        if n.note_n == note && n.on {
                            n.on = false;
                            n.asdr.release_level = n.asdr.attack_level;
                        }
                    })
                }
            },
            (),
        )
        .unwrap();

    // AUDIO
    let host = cpal::default_host();
    let dev = host.default_output_device().unwrap();
    let mut config: cpal::StreamConfig = dev.default_output_config().unwrap().into();
    config.buffer_size = BufferSize::Fixed(256);
    let sample_rate = config.sample_rate.0 as f32;
    println!("{config:?}");

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                fn apply_notes_to_single_sample(
                    d: &mut f32,
                    pan_mul: f32,
                    pan_offset: f32,
                    notes: &mut [Note],
                    sample_rate: f32,
                ) {
                    let dt = 1.0 / sample_rate;
                    *d = 0.0;
                    for n in notes.iter_mut() {
                        n.phase += n.freq / 20.0;
                        if n.phase as usize > n.samples_at_20_hz.len() - 1 {
                            n.phase -= n.samples_at_20_hz.len() as f32
                        }
                        let idx = n.phase as usize;
                        let sample = n.samples_at_20_hz[idx] * n.vel;
                        let panned = sample * (pan_offset + n.pan * pan_mul);
                        let asdrd = panned
                            * if n.on {
                                n.asdr.attack(dt)
                            } else {
                                n.asdr.release(dt)
                            };

                        *d += asdrd;
                    }
                }
                let mut notes = notes2.lock().unwrap();
                for c in data.chunks_exact_mut(2) {
                    // left
                    apply_notes_to_single_sample(&mut c[0], -1.0, 1.0, &mut notes, sample_rate);
                    // right
                    apply_notes_to_single_sample(&mut c[1], 1.0, 0.0, &mut notes, sample_rate)
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
                notes: notes3,
                settings: settings_writer,
            })
        }),
    )
    .unwrap();
}

struct App {
    vis_buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    notes: Arc<Mutex<Vec<Note>>>,
    settings: Arc<Settings>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        {
            self.notes
                .lock()
                .unwrap()
                .retain_mut(|n| n.asdr.release(0.0) > 0.01)
        }
        let mut buf = self.vis_buf.lock().unwrap();
        let to_remove = buf.len().saturating_sub(VIS_BUF_SIZE);
        for _ in 0..to_remove {
            buf.pop_front();
        }

        let step_by = 20;
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
            slider(
                ui,
                &self.settings.fat_phase_offset,
                -30000..=30000,
                "phase offset",
            );
            slider(ui, &self.settings.fat_detune, -10000..=10000, "detune");
            slider(ui, &self.settings.fat_n, 1..=100, "chorun N");
            slider(ui, &self.settings.fat_pan, -10000..=10000, "chorun pan");
            slider(ui, &self.settings.boost, 0..=100000, "boost");

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
                    let live: PlotPoints = buf
                        .iter()
                        .enumerate()
                        .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                        .step_by(step_by)
                        .collect();

                    let locked: PlotPoints = self
                        .locked
                        .iter()
                        .enumerate()
                        .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                        .collect();

                    ui.line(Line::new(live));
                    ui.line(Line::new(locked).name("locked"));
                });
        });
    }
}
