use std::{
    collections::VecDeque,
    f32::consts::{PI, TAU},
    sync::{Arc, Mutex},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use eframe::egui::CentralPanel;
use egui_plot::{Line, Plot, PlotPoints};
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;

#[derive(Debug)]
struct Note {
    note_n: u8,
    phase: f32,
    freq: f32,
    vel: f32,
    asdr: Asdr,
    on: bool,
}

#[derive(Debug)]
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

const VIS_BUF_SIZE: usize = 44100 * 3;

fn main() {
    // FIXME: opt
    let notes = Arc::new(Mutex::new(Vec::<Note>::new()));
    let notes2 = Arc::clone(&notes);
    let notes3 = Arc::clone(&notes);

    let vis_buf = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buf2 = Arc::clone(&vis_buf);

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
                    };
                    note_counter += 1;
                    println!("{note_counter} {note:?}");
                    notes.lock().unwrap().push(note)
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
    println!("{sample_rate:?}");

    let volume = 0.9;

    let stream = dev
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                for p in data.iter_mut() {
                    *p = 0.0;
                    let mut notes = notes2.lock().unwrap();
                    for n in notes.iter_mut() {
                        n.phase += n.freq * TAU * dt;
                        if n.phase > PI {
                            n.phase -= TAU;
                        }
                        *p += n.phase.sin()
                            * n.vel
                            * volume
                            * if n.on {
                                n.asdr.attack(dt)
                            } else {
                                n.asdr.release(dt)
                            }
                    }
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

    eframe::run_native(
        "Riano",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(App::new(buf2, notes3))),
    )
    .unwrap();
}

struct App {
    buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
    notes: Arc<Mutex<Vec<Note>>>,
}

impl App {
    fn new(buf: Arc<Mutex<VecDeque<f32>>>, notes: Arc<Mutex<Vec<Note>>>) -> Self {
        Self {
            buf,
            locked: vec![],
            notes,
        }
    }
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
        let mut buf = self.buf.lock().unwrap();
        let to_remove = buf.len().saturating_sub(VIS_BUF_SIZE);
        let mut remainder = buf.split_off(to_remove);
        std::mem::swap(&mut remainder, &mut buf);
        CentralPanel::default().show(ctx, |ui| {
            if ui.button("lock").clicked() {
                self.locked = buf.iter().copied().collect();
            }
            if ui.button("reset").clicked() {
                self.locked.clear();
            }
            Plot::new("vaweform").show(ui, |ui| {
                let live: PlotPoints = buf
                    .iter()
                    .enumerate()
                    .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                    .step_by(10)
                    .collect();

                let locked: PlotPoints = self
                    .locked
                    .iter()
                    .enumerate()
                    .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
                    .step_by(10)
                    .collect();

                ui.line(Line::new(live));
                ui.line(Line::new(locked).name("locked"));
            });
        });
    }
}
