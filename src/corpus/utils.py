import pandas as pd
import numpy as np
import pretty_midi

def print_data(df: pd.DataFrame) -> None:
    print("Data: ")
    print(df)
    # Get basic info (useful for checking data types, non-null counts)
    print("\nDataFrame Info:")
    df.info()

# AI generated function
def split_pretty_midi_into_segments(midi, segment_duration: float):
    ret = []
    total_midi_duration = midi.get_end_time()
    num_segments = int(np.ceil(total_midi_duration / segment_duration))

    for i in range(num_segments):
        segment_start_time = i * segment_duration
        segment_end_time = min(segment_start_time + segment_duration, total_midi_duration + 0.001) # Add small buffer

        segment_pm = pretty_midi.PrettyMIDI()

        # Copy global meta-events (like tempo changes, time signatures)
        # and adjust their times relative to the segment start.
        # This is important for accurate playback/interpretation of segments.
        for tempo_change_time, tempo in midi.get_tempo_changes():
            if segment_start_time <= tempo_change_time < segment_end_time:
                segment_pm.adjust_times([(tempo_change_time - segment_start_time, tempo)])

        for time_signature in midi.time_signature_changes:
            if segment_start_time <= time_signature.time < segment_end_time:
                new_ts = pretty_midi.TimeSignature(
                    numerator=time_signature.numerator,
                    denominator=time_signature.denominator,
                    time=time_signature.time - segment_start_time
                )
                segment_pm.time_signature_changes.append(new_ts)
        
        # Iterate through instruments and notes
        for original_instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=original_instrument.program,
                is_drum=original_instrument.is_drum,
                name=original_instrument.name
            )
            
            for note in original_instrument.notes:
                # Check if the note overlaps with the current segment
                # A note overlaps if its start is before the segment ends AND its end is after the segment starts
                if note.start < segment_end_time and note.end > segment_start_time:
                    # Adjust note start and end times relative to the segment's beginning
                    adjusted_start = max(0.0, note.start - segment_start_time)
                    adjusted_end = min(segment_duration, note.end - segment_start_time)

                    # Only add the note if it still has a positive duration within the segment
                    if adjusted_end > adjusted_start:
                        new_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=adjusted_start,
                            end=adjusted_end
                        )
                        new_instrument.notes.append(new_note)
            
            # Only add the instrument to the segment_pm if it contains notes in this segment
            if new_instrument.notes:
                segment_pm.instruments.append(new_instrument)
        
        ret.append(segment_pm)
        print(f"  Segment {i+1}/{num_segments} (Time: {segment_start_time:.2f}s to {segment_end_time:.2f}s) created. Notes: {sum(len(inst.notes) for inst in segment_pm.instruments)}")

    return ret
