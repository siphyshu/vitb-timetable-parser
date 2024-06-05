import streamlit as st
import base64
from ocr import parse_timetable
from utils import convert_timetable_to_json


def main():
    st.title("vitb-timetable-parser")
    st.markdown("""This is a demo web app that parses vitb timetables images to json/csv using [vitb-timetable-parser](https://github.com/siphyshu/vitb-timetable-parser).""")
    st.divider()

    
    uploaded_file = st.file_uploader("Upload your timetable", type=["png", "jpeg"], accept_multiple_files=False, help="Upload a clear and high-quality __screenshot__ of the timetable. Make sure the timetable is not cropped or rotated.")

    if uploaded_file is not None:
        # Display progress bar or success message in status container
        status_container = st.empty()
        status_container2 = st.empty()
        progress_bar = status_container.progress(0)

        # Display uploaded image
        st.image(uploaded_file, use_column_width=True)

        try:
            # Parse timetable
            parsed_timetable = parse_timetable(image=uploaded_file, progress_callback=lambda progress: progress_bar.progress(progress))
        except ValueError as e:
            # Show error message in status container
            status_container.error(str(e))
            return
        else:
            # Show success message in status container
            status_container.success("Timetable extracted from image successfully!")
            
            # Display parsed timetable
            st.write("Parsed Timetable:")
            st.dataframe(parsed_timetable)
        
        try: 
            # Convert timetable to JSON
            timetable_json = convert_timetable_to_json(parsed_timetable)
        except ValueError as e:
            # Add error message to status container
            status_container2.error("Failed to convert timetable to JSON format.")
        else:
            # Add success message to status container
            status_container2.success("Timetable converted to JSON successfully!")
            # Display JSON download link
            st.write("JSON Representation:")
            st.json(timetable_json)


if __name__ == "__main__":
    main()