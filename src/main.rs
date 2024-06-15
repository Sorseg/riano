use std::{
    collections::VecDeque,
    f32::consts::{PI, TAU},
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc, Mutex,
    },
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use eframe::egui::{CentralPanel, Slider, Ui, Vec2b};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;

#[derive(Debug, Clone)]
struct Note {
    note_n: u8,
    phase: f32,
    freq: f32,
    vel: f32,
    asdr: Asdr,
    on: bool,
    /// 0.0 - left, 1.0 - right
    pan: f32,
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

#[derive(Debug, Default)]
struct Settings {
    chorus_phase_offset: AtomicI32,
    fat_detune: AtomicI32
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
                    };

                    let fat_n = 4;

                    for fat in 0..fat_n {
                        // -0.5 .. 0.5
                        let fat_offset = (fat as f32 - fat_n as f32 / 2.0) / fat_n as f32;
                        let mut note = note.clone();
                        note.freq *= 1.0 + fat as f32 * (settings_reader.fat_detune.load(Ordering::Relaxed) as f32 / 10000.0);
                        note.pan = 0.5 + fat_offset * 0.2;
                        note.vel *= (fat_n - fat) as f32 / fat_n as f32;
                        note.phase = (fat as f32 / fat_n as f32)
                            * (settings_reader.chorus_phase_offset.load(Ordering::Relaxed) as f32
                                / 10000.0);

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
    let dt = 1.0 / sample_rate;
    println!("{config:?}");

    let volume = 0.8;

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let mut notes = notes2.lock().unwrap();
                let mut apply_notes = |d: &mut f32, pan_inv: bool| {
                    *d = 0.0;
                    for n in notes.iter_mut() {
                        n.phase += n.freq * TAU * dt;
                        if n.phase > PI {
                            n.phase -= TAU;
                        }
                        *d += n.phase.sin().powf(5.0)
                            * n.vel
                            * volume
                            * if pan_inv { 1.0 - n.pan } else { n.pan }
                            * if n.on {
                                n.asdr.attack(dt)
                            } else {
                                n.asdr.release(dt)
                            }
                    }
                };
                for c in data.chunks_exact_mut(2) {
                    // left
                    apply_notes(&mut c[0], true);
                    // right
                    apply_notes(&mut c[1], false)
                }

                let mut vis_buf = vis_buf.lock().unwrap();
                vis_buf.extend(data.iter().copied());
                if vis_buf.len() > VIS_BUF_SIZE * 2 {
                    panic!("buf not consumed");
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

            let slider = |ui: &mut Ui, atomic: &AtomicI32, range, text: &str|{
                let mut val = atomic.load(Ordering::Relaxed);
                let prev_value = val;
                ui.add(Slider::new(&mut val, range).text(text));
                if val != prev_value {
                    atomic.store(val, Ordering::Relaxed);
                }
            };
            slider(ui, &self.settings.chorus_phase_offset, -30000..=30000, "phase offset");
            slider(ui, &self.settings.fat_detune, -10000..=10000, "detune");

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
