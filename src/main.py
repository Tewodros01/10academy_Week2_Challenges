# src/main.py
import streamlit as st
from utils import *

def main():
    st.title("Telecom Data Analysis Dashboard")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Task 1", "Task 2", "Task 3", "Task 4"))

    if page == "Task 1":
        task_1()
    elif page == "Task 2":
        task_2()
    elif page == "Task 3":
        task_3()
    elif page == "Task 4":
        task_4()

if __name__ == "__main__":
    main()
