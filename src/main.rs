use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    sync::{
        atomic::{AtomicU16, AtomicU8, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize,
};
use midi_msg::{ChannelVoiceMsg, MidiMsg};
use midir::MidiInput;

#[derive(Debug)]
struct Note {
    note_n: u8,
    phase: f32,
    freq: f32,
    vel: f32,
}

fn main() {
    // FIXME: opt
    let notes = Arc::new(Mutex::new(Vec::<Note>::new()));
    let notes2 = Arc::clone(&notes);

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
                for p in data {
                    let mut notes = notes2.lock().unwrap();
                    if notes.is_empty() {
                        *p = 0.0;
                    }
                    for n in notes.iter_mut() {
                        n.phase += n.freq / 44100.0;
                        if n.phase > PI {
                            n.phase -= TAU;
                        }
                        *p += n.phase.sin() * n.vel;
                    }
                }
            },
            |err| eprintln!("{err:?}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();

    // SLEEP
    loop {
        std::thread::sleep(Duration::from_secs(10));
    }
}
