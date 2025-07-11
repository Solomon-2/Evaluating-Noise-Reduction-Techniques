import xml.etree.ElementTree as ET
import csv

def extract_apnea_events(xml_file_path, output_csv=None):
    """
    Extract apnea events from the patient XML file.
    
    Args:
        xml_file_path: Path to the XML file
        output_csv: Optional path to save results as CSV
    
    Returns:
        List of tuples (event_type, start_time, end_time)
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Define the namespace (from your XML file)
    namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
    
    apnea_events = []
    
    # Find all Event elements
    for event in root.findall('.//ns:Event', namespace):
        event_family = event.get('Family')
        event_type = event.get('Type')
        
        # Check if this is a respiratory apnea event
        if (event_family == 'Respiratory' and 
            event_type in ['ObstructiveApnea', 'CentralApnea', 'MixedApnea']):
            
            start_time = float(event.get('Start'))
            duration = float(event.get('Duration'))
            end_time = start_time + duration
            
            apnea_events.append((event_type, start_time, end_time))
            print(f"{event_type}: {start_time}s to {end_time}s (duration: {duration}s)")
    
    # Sort events by start time
    apnea_events.sort(key=lambda x: x[1])
    
    # Save to CSV if requested
    if output_csv:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['event_type', 'start_sec', 'end_sec'])
            for event_type, start, end in apnea_events:
                writer.writerow([event_type, start, end])
        print(f"\nApnea events saved to {output_csv}")
    
    return apnea_events

# Usage example
if __name__ == "__main__":
    xml_path = "../data/patient_description.rml"
    
    # Extract apnea events
    events = extract_apnea_events(xml_path, "apnea_ground_truth.csv")
    
    print(f"\nFound {len(events)} apnea events total")
    
    # Count by type
    event_counts = {}
    for event_type, _, _ in events:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print("\nEvent counts by type:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")