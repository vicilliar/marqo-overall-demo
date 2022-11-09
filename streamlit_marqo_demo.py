import os
import streamlit as st
import pandas as pd
import marqo

from pprint import pprint
from PIL import Image
from marqo.errors import IndexAlreadyExistsError, IndexNotFoundError

# constants
CSV_DATASET = "airwallex.csv"
CSV_HEADER = ["url", "title", "body", "scraped_from"]
INDEX_NAME = "airwallex-v4mpnetbase"
DOCKER_INTERNAL = "http://host.docker.internal:8222/"
DOCKER_CONTAINER_URL = "http://localhost:8882"
EN_GB = "English"
ZH_CN = "Chinese"
SEARCH_LANG_MODE_OPTIONS = (EN_GB, )  # ZH_CN)
TENSOR_SEARCH_MODE = "TENSOR"
LEXICAL_SEARCH_MODE = "LEXICAL"
SEARCH_TEXT_MODE_OPTIONS = ("Tensor", "Lexical")
PRE_FILTERING_OPTIONS = [
    "faq",
    "blogs",
    "landing",
    "newsroom",
]
SEARCHABLE_ATTRS = ["title", "body", "scraped_from"]
RESULT_TABLE_HEADERS = [
    {"key_no": 0, "title": "Result No."},
    {"key_no": 1, "title": "Details"},
    {"key_no": 2, "title": "Score"},
]

# Streamlit configuration settings
st.set_page_config(
    page_title="Marqo Demo App",
    page_icon="favicon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={}
)

mq = marqo.Client(url=DOCKER_CONTAINER_URL)  # Connection to Marqo Docker Container
cwd = os.getcwd()  # Get current working directory


def reset_state():
    st.session_state["results"] = {}
    st.session_state["page"] = -1


def load_index(number_data):
    try:
        articles_dataset = pd.read_csv(CSV_DATASET).head(number_data)[CSV_HEADER].to_dict("records")
        settings = {
            "treat_urls_and_pointers_as_images": False,  # allows us to find an image file and index it
            "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            # "normalize_embeddings": False,
        }

        mq.create_index(INDEX_NAME, **settings)

        with st.spinner("Creating Index..."):
            mq.index(INDEX_NAME).add_documents(articles_dataset, batch_size=100)

        st.success("Index successfully created.")
    except IndexAlreadyExistsError:
        st.error("Index already exists.")


def delete_index():
    try:
        mq.index(INDEX_NAME).delete()
        st.success("Index successfully deleted.")
    except IndexNotFoundError:
        st.error("Index does not exist.")


def render_index_settings_ui():
    # Index Settings Frontend
    with st.sidebar:
        st.write("Index Settings:")
        values = st.slider(
            label="Select a range of values",
            min_value=10.0,
            max_value=5000.0,
            value=1000.0,
            step=10.0)

        create_col, _, delete_col = st.columns([1, 1, 1])

        with create_col:
            create_btn = st.button("Create Index")
        if create_btn:
            load_index(int(values))
        with delete_col:
            delete_btn = st.button("Delete Index")
        if delete_btn:
            delete_index()


def get_search_method(search_text: str | None):
    word_count = len(search_text.split(" "))
    search_text_mode = TENSOR_SEARCH_MODE if word_count > 1 else LEXICAL_SEARCH_MODE
    return search_text_mode


def render_main_app_ui():
    # Main application frontend
    logo = Image.open(f"{cwd}/marqo-logo.jpg")
    st.image(logo)

    search_text, search_article_url = None, None
    search_mode = st.radio("Language", SEARCH_LANG_MODE_OPTIONS, horizontal=True, on_change=reset_state)

    if search_mode == EN_GB:
        search_text = st.text_input("Text Search")
        st.write(f"Search mode: {get_search_method(search_text).capitalize()}")

    with st.expander("Search Settings"):
        attr_col, filter_col = st.columns(2)
        with attr_col:
            searchable_attr = st.multiselect("Searchable Attributes", SEARCHABLE_ATTRS, default=SEARCHABLE_ATTRS)

        with filter_col:
            filtering = st.multiselect("Pre-filtering Options", PRE_FILTERING_OPTIONS, default=PRE_FILTERING_OPTIONS)

    return [
        search_text,
        searchable_attr,
        filtering,
    ]


def render_result_rows(search_method: str):
    for row in enumerate(st.session_state["results"]["hits"]):
        url = row[1].get("url", "")
        title = row[1].get("title", "")
        body = row[1].get("body", "")
        score = row[1].get("_score", "")
        highlights = row[1].get("_highlights", {})

        if type(highlights) == list:
            readable_highlights = None
        else:
            readable_highlights = list(highlights.values())[0] if search_method == TENSOR_SEARCH_MODE else None

        # row contains the following:
        #   result#, {title, body, url, _id, _highlights: {body or title}, _score}
        if (row[0] >= st.session_state["page"] * 10) and (row[0] < (st.session_state["page"] * 10 + 10)):
            with st.expander(f"{row[0] + 1} - {title}", expanded=True):  # wraps the whole row
                st.info(f"Score: {score}\n\n**Highlights:** {readable_highlights}\n\n **[Go to page]({url})**", icon="ℹ️")
                st.write(body)


def render_results_pagination(search_method: str):
    # Results Pagination Logic
    if st.session_state["page"] > -1:
        prev_col, page_col, next_col = st.columns([1, 9, 1])

        with prev_col:
            prev_btn = st.button("Prev")
            if prev_btn and (st.session_state["page"] > 0):
                st.session_state["page"] -= 1

        with next_col:
            next_btn = st.button("Next")
            if next_btn and (st.session_state["page"] < 2):
                st.session_state["page"] += 1

        with page_col:
            page_no = str(st.session_state["page"] + 1)
            st.markdown(f"<div style='text-align: center'> {page_no}</div>", unsafe_allow_html=True)

    if st.session_state["results"] != {}:
        if st.session_state["results"]["hits"]:
            st.write("Results (Top 30):")
            render_result_rows(search_method)
        else:
            st.write("No results")


def create_filter_str(filter_list):
    filter_string = ""

    for field in filter_list:
        filter_string += f"{' OR ' if filter_string != '' else ''}scraped_from:({field})"

    return filter_string


def render_results(search_text, searchable_attr, search_btn, filtering):
    # Marqo Results logic
    search_method = get_search_method(search_text)

    if search_text != "" and search_text is not None and search_btn:
        results = mq.index(INDEX_NAME).search(
            search_text,
            filter_string=create_filter_str(filtering),
            search_method=search_method,
            searchable_attributes=[i.lower() for i in searchable_attr],
            limit=30
        )

        st.session_state["results"] = results

        if st.session_state["results"]["hits"]:
            st.session_state["page"] = 0
        else:
            st.session_state["page"] = -1

    render_results_pagination(search_method)


def main():
    # Streamlit state variables (this is to save the state of the session for pagination of Marqo query results)
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    if "page" not in st.session_state:
        st.session_state["page"] = -1

    render_index_settings_ui()
    search_text, searchable_attr, filtering = render_main_app_ui()
    search_btn = st.button("Search")
    render_results(search_text, searchable_attr, search_btn, filtering)


if __name__ == "__main__":
    main()
