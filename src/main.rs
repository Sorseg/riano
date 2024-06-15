use std::{
    collections::VecDeque, f32::consts::{PI, TAU}, sync::{Arc, Mutex}
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
}

const VIS_BUF_SIZE: usize = 44100 * 1;

fn main() {
    // FIXME: opt
    let notes = Arc::new(Mutex::new(Vec::<Note>::new()));
    let notes2 = Arc::clone(&notes);

    let buf = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buf2 = Arc::clone(&buf);

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
                    let note = Note {
                        note_n: note,
                        freq: (note - 21) as f32 * 10.0,
                        vel: velocity as f32 / 255.0,
                        phase: 0.0,
                    };
                    println!("note {note:?}");
                    notes.lock().unwrap().push(note)
                }
                if let MidiMsg::ChannelVoice {
                    msg: ChannelVoiceMsg::NoteOff { note, .. },
                    ..
                } = m
                {
                    notes.lock().unwrap().retain(|n| n.note_n != note)
                }
            },
            (),
        )
        .unwrap();

    // AUDIO
    let host = cpal::default_host();
    let dev = host.default_output_device().unwrap();
    let mut config: cpal::StreamConfig = dev.default_output_config().unwrap().into();
    config.buffer_size = BufferSize::Fixed(512);

    let stream = dev
        .build_output_stream(
            &config.into(),
            move |data: &mut [f32], _| {
                for p in data.iter_mut() {
                    *p = 0.0;
                    let mut notes = notes2.lock().unwrap();
                    for n in notes.iter_mut() {
                        n.phase += n.freq / 44100.0;
                        if n.phase > PI {
                            n.phase -= TAU;
                        }
                        *p += n.phase.sin() as f32 * n.vel;
                    }
                }
                let mut buf = buf.lock().unwrap();
                buf.extend(data.iter().copied());
                if buf.len() > VIS_BUF_SIZE * 2 {
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
        Box::new(|_cc| Box::new(App::new(buf2))),
    )
    .unwrap();
}

struct App {
    buf: Arc<Mutex<VecDeque<f32>>>,
    locked: Vec<f32>,
}

impl App {
    fn new(buf: Arc<Mutex<VecDeque<f32>>>) -> Self {
        Self {
            buf,
            locked: vec![],
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let mut buf = self.buf.lock().unwrap();
        let to_remove = buf.len().saturating_sub(VIS_BUF_SIZE);
        let mut remainder = buf.split_off(to_remove);
        std::mem::swap(&mut remainder, &mut buf);
        CentralPanel::default().show(ctx, |ui| {
            if ui.button("lock").clicked() {
                self.locked = buf.iter().copied().collect();
            }
            Plot::new("vaweform").show(ui, |ui| {
                let live: PlotPoints = buf
                    .iter()
                    .enumerate()
                    .map(|(i, s)| [i as f64 / 44100.0, *s as f64])
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
