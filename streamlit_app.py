import streamlit as st
import base64
from main import parse_timetable


def get_table_download_link(df, file_type, text):
    """
    Generate a link to download the DataFrame as a CSV file.
    """
    if file_type == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="parsed_timetable.csv">{text}</a>'
    elif file_type == 'json':
        json = df.to_json(orient='records')
        b64 = base64.b64encode(json.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="parsed_timetable.json">{text}</a>'
    return href


def main():
    st.title("vitb-timetable-parser")
    # add chip to github repo https://github.com/siphyshu/vitb-timetable-parser
    st.markdown("""This is a demo web app that parses vitb timetables images to json/csv using [vitb-timetable-parser](https://github.com/siphyshu/vitb-timetable-parser).""")
    st.divider()

    # File upload
    uploaded_file = st.file_uploader("Upload your timetable", type=["png", "jpeg"], accept_multiple_files=False)

    if uploaded_file is not None:
        # Display uploaded image
        st.write("Uploaded Timetable:")
        st.image(uploaded_file, use_column_width=True)

        # Parse timetable
        parsed_timetable = parse_timetable(image=uploaded_file)

        # Display parsed timetable
        st.write("Parsed Timetable:")
        st.dataframe(parsed_timetable)


if __name__ == "__main__":
    main()