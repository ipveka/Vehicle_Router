"""
UI Components Module

This module provides reusable UI components and widgets for the
Streamlit application.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional


class UIComponents:
    """
    UI Components for Streamlit Application
    
    This class provides reusable UI components and widgets for the
    Vehicle Router Streamlit application.
    """
    
    def __init__(self):
        """Initialize the UIComponents"""
        pass
    
    def create_metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                          help_text: Optional[str] = None):
        """
        Create a styled metric card
        
        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta value
            help_text: Optional help text
        """
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
    
    def create_info_box(self, title: str, content: str, box_type: str = "info"):
        """
        Create an information box with custom styling
        
        Args:
            title: Box title
            content: Box content
            box_type: Type of box (info, success, warning, error)
        """
        if box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
        else:
            st.info(f"**{title}**\n\n{content}")
    
    def create_progress_tracker(self, steps: List[str], current_step: int):
        """
        Create a progress tracker showing current step
        
        Args:
            steps: List of step names
            current_step: Current step index (0-based)
        """
        progress = (current_step + 1) / len(steps)
        st.progress(progress)
        
        # Show current step
        if current_step < len(steps):
            st.write(f"**Current Step:** {steps[current_step]}")
        
        # Show all steps with status
        for i, step in enumerate(steps):
            if i < current_step:
                st.write(f"✅ {step}")
            elif i == current_step:
                st.write(f"🔄 {step}")
            else:
                st.write(f"⏳ {step}")
    
    def create_data_table(self, df: pd.DataFrame, title: str, 
                         column_config: Optional[Dict] = None,
                         height: Optional[int] = None):
        """
        Create a styled data table
        
        Args:
            df: DataFrame to display
            title: Table title
            column_config: Optional column configuration
            height: Optional table height
        """
        st.subheader(title)
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config,
            height=height
        )
    
    def create_download_section(self, data: bytes, filename: str, 
                               mime_type: str, button_text: str):
        """
        Create a download section with button
        
        Args:
            data: Data to download
            filename: Download filename
            mime_type: MIME type
            button_text: Button text
        """
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type,
            type="primary"
        )
    
    def create_status_indicator(self, status: str, message: str):
        """
        Create a status indicator
        
        Args:
            status: Status type (success, warning, error, info)
            message: Status message
        """
        if status == "success":
            st.success(f"✅ {message}")
        elif status == "warning":
            st.warning(f"⚠️ {message}")
        elif status == "error":
            st.error(f"❌ {message}")
        else:
            st.info(f"ℹ️ {message}")
    
    def create_expandable_section(self, title: str, content: str, expanded: bool = False):
        """
        Create an expandable section
        
        Args:
            title: Section title
            content: Section content
            expanded: Whether section is expanded by default
        """
        with st.expander(title, expanded=expanded):
            st.write(content)
    
    def create_sidebar_section(self, title: str, content_func):
        """
        Create a sidebar section
        
        Args:
            title: Section title
            content_func: Function to render section content
        """
        st.sidebar.markdown(f"### {title}")
        content_func()
    
    def create_tabs_section(self, tab_names: List[str], tab_contents: List[callable]):
        """
        Create a tabbed section
        
        Args:
            tab_names: List of tab names
            tab_contents: List of functions to render tab content
        """
        tabs = st.tabs(tab_names)
        
        for tab, content_func in zip(tabs, tab_contents):
            with tab:
                content_func()
    
    def create_two_column_layout(self, left_content: callable, right_content: callable,
                                col_ratio: tuple = (1, 1)):
        """
        Create a two-column layout
        
        Args:
            left_content: Function to render left column content
            right_content: Function to render right column content
            col_ratio: Column ratio tuple
        """
        col1, col2 = st.columns(col_ratio)
        
        with col1:
            left_content()
        
        with col2:
            right_content()
    
    def create_three_column_layout(self, left_content: callable, center_content: callable,
                                  right_content: callable, col_ratio: tuple = (1, 1, 1)):
        """
        Create a three-column layout
        
        Args:
            left_content: Function to render left column content
            center_content: Function to render center column content
            right_content: Function to render right column content
            col_ratio: Column ratio tuple
        """
        col1, col2, col3 = st.columns(col_ratio)
        
        with col1:
            left_content()
        
        with col2:
            center_content()
        
        with col3:
            right_content()
    
    def create_loading_spinner(self, message: str = "Loading..."):
        """
        Create a loading spinner context manager
        
        Args:
            message: Loading message
            
        Returns:
            Context manager for loading spinner
        """
        return st.spinner(message)
    
    def create_form_section(self, form_key: str, submit_text: str = "Submit"):
        """
        Create a form section context manager
        
        Args:
            form_key: Unique form key
            submit_text: Submit button text
            
        Returns:
            Context manager for form
        """
        return st.form(form_key)
    
    def create_alert_box(self, message: str, alert_type: str = "info"):
        """
        Create an alert box
        
        Args:
            message: Alert message
            alert_type: Type of alert (info, success, warning, error)
        """
        if alert_type == "success":
            st.success(message)
        elif alert_type == "warning":
            st.warning(message)
        elif alert_type == "error":
            st.error(message)
        else:
            st.info(message)