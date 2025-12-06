import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np

from config import (
    APP_TITLE, STATES, INCIDENT_TYPES, DECLARATION_TYPES, ESF_COLUMNS,
    PROBABILITY_THRESHOLDS, ERROR_MESSAGES, ESF_MODEL_PATH, AR_TOPIC_MODEL_PATH,
    AR_TOPICS_FILE, SOW_TOPICS_FILE, STATE_ABBREV
)
from create_input_stage1 import stage_one_input
from cap_search import build_index
from create_input_stage2 import get_AR_topics, get_SOW_topics


@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[object]]:
    """Load ESF and AR topic models from disk.
    
    Returns:
        Tuple containing (esf_model, ar_topic_model)
    """
    esf_model = None
    ar_topic_model = None
    
    try:
        if ESF_MODEL_PATH.exists():
            with open(ESF_MODEL_PATH, 'rb') as f:
                esf_model = pickle.load(f)
        else:
            st.error(f"{ERROR_MESSAGES['file_not_found']}: {ESF_MODEL_PATH}")
    except Exception as e:
        st.error(f"{ERROR_MESSAGES['model_load_error']}: ESF model - {e}")
    
    try:
        if AR_TOPIC_MODEL_PATH.exists():
            with open(AR_TOPIC_MODEL_PATH, 'rb') as f:
                ar_topic_model = pickle.load(f)
        else:
            st.error(f"{ERROR_MESSAGES['file_not_found']}: {AR_TOPIC_MODEL_PATH}")
    except Exception as e:
        st.error(f"{ERROR_MESSAGES['model_load_error']}: AR topic model - {e}")
    
    return esf_model, ar_topic_model


def categorize_probabilities(probabilities: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
    """Group ESF predictions into probability categories.
    
    Args:
        probabilities: ESF probability predictions
        
    Returns:
        Dictionary with keys: highly_likely, likely, unlikely, very_unlikely
    """
    esf_df = pd.DataFrame(
        probabilities.reshape(1, -1),
        columns=ESF_COLUMNS
    )
    
    categories = {
        'highly_likely': [],
        'likely': [],
        'unlikely': [],
        'very_unlikely': []
    }
    
    for esf, prob_list in esf_df.to_dict('list').items():
        prob = prob_list[0]
        
        if prob > PROBABILITY_THRESHOLDS['highly_likely']:
            categories['highly_likely'].append((esf, prob))
        elif prob >= PROBABILITY_THRESHOLDS['likely']:
            categories['likely'].append((esf, prob))
        elif prob >= PROBABILITY_THRESHOLDS['unlikely']:
            categories['unlikely'].append((esf, prob))
        else:
            categories['very_unlikely'].append((esf, prob))
    
    return categories


def display_esf_predictions(categories: Dict[str, List[Tuple[str, float]]]) -> None:
    """Display ESF predictions with checkboxes in four probability columns.
    
    Args:
        categories: Categorized ESF predictions
    """
    if 'selected_esfs' not in st.session_state:
        st.session_state['selected_esfs'] = []
    
    col_header1, col_header2 = st.columns([3, 1])
    with col_header2:
        if st.button('🗑️ Clear All Selections', use_container_width=True, key='clear_esf_btn'):
            st.session_state['selected_esfs'] = []
            st.rerun()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('**Highly Likely** :green[(>75%)]')
        if categories['highly_likely']:
            for esf, prob in categories['highly_likely']:
                is_checked = esf in st.session_state['selected_esfs']
                if st.checkbox(f"{esf} ({prob:.1%})", value=is_checked, key=f"check_{esf}_high"):
                    if esf not in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].append(esf)
                else:
                    if esf in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].remove(esf)
        else:
            st.info("None")
    
    with col2:
        st.markdown('**Likely** :blue[(50-75%)]')
        if categories['likely']:
            for esf, prob in categories['likely']:
                is_checked = esf in st.session_state['selected_esfs']
                if st.checkbox(f"{esf} ({prob:.1%})", value=is_checked, key=f"check_{esf}_mid"):
                    if esf not in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].append(esf)
                else:
                    if esf in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].remove(esf)
        else:
            st.info("None")
    
    with col3:
        st.markdown('**Unlikely** :orange[(25-50%)]')
        if categories['unlikely']:
            for esf, prob in categories['unlikely']:
                is_checked = esf in st.session_state['selected_esfs']
                if st.checkbox(f"{esf} ({prob:.1%})", value=is_checked, key=f"check_{esf}_low"):
                    if esf not in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].append(esf)
                else:
                    if esf in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].remove(esf)
        else:
            st.info("None")
    
    with col4:
        st.markdown('**Very Unlikely** :gray[(<25%)]')
        if categories['very_unlikely']:
            for esf, prob in categories['very_unlikely']:
                is_checked = esf in st.session_state['selected_esfs']
                if st.checkbox(f"{esf} ({prob:.1%})", value=is_checked, key=f"check_{esf}_vlow"):
                    if esf not in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].append(esf)
                else:
                    if esf in st.session_state['selected_esfs']:
                        st.session_state['selected_esfs'].remove(esf)
        else:
            st.info("None")


def display_ar_topics(min_probability: float = 0.10) -> None:
    """Display AR topics with matching SOW statements.
    
    Args:
        min_probability: Minimum probability threshold for topic display
    """
    try:
        if 'ar_topic_probs' not in st.session_state:
            st.warning("No AR topic predictions available. Please generate suggestions first.")
            return
        
        if 'selected_sows' not in st.session_state:
            st.session_state['selected_sows'] = {}
        if 'temp_sow_selections' not in st.session_state:
            st.session_state['temp_sow_selections'] = {}
        if 'sow_confirmed' not in st.session_state:
            st.session_state['sow_confirmed'] = False
        
        if st.session_state['sow_confirmed']:
            st.success(f"✓ {len(st.session_state['selected_sows'])} Statement(s) of Work confirmed")
            
            if st.session_state['selected_sows']:
                with st.expander("📋 View Confirmed Selections", expanded=False):
                    for sow_key, sow_data in st.session_state['selected_sows'].items():
                        st.markdown(f"**{sow_data['ar_topic']}** ({sow_data['probability']})")
                        for item in sow_data['sow_items']:
                            st.markdown(f"  - {item}")
                        st.divider()
            
            st.info("💡 To modify selections, click 'Suggest Assistance Request Topics' again.")
            return
        
        topic_probs = st.session_state['ar_topic_probs']
        ar_topics = get_AR_topics(str(AR_TOPICS_FILE))
        
        if topic_probs:
            if 'sow_search_results' not in st.session_state:
                loading_placeholder = st.empty()
                loading_placeholder.info('🔍 Loading Statements of Work... Please wait.')
                
                sow_index, sow_topics = build_index(str(SOW_TOPICS_FILE))
                st.session_state['sow_search_results'] = {}
                
                if sow_index and sow_topics:
                    from cap_search import _tokenize_and_stem
                    
                    for topic_id, prob in topic_probs:
                        if topic_id in ar_topics:
                            topic_text = ar_topics[topic_id]
                            query_tokens = _tokenize_and_stem(topic_text)
                            
                            if query_tokens:
                                accumulator = {}
                                for token in query_tokens:
                                    if token in sow_index:
                                        for doc_id in sow_index[token]:
                                            accumulator[doc_id] = accumulator.get(doc_id, 0) + 1
                                
                                if accumulator:
                                    sorted_docs = sorted(
                                        accumulator.items(),
                                        key=lambda item: item[1],
                                        reverse=True
                                    )
                                    
                                    results = []
                                    seen = set()
                                    for doc_id, _ in sorted_docs:
                                        if doc_id in sow_topics:
                                            sow_text = sow_topics[doc_id].strip()
                                            if ' - ' in sow_text:
                                                parts = sow_text.split(' - ', 1)
                                                if parts[0].strip().isdigit():
                                                    sow_text = parts[1].strip()
                                            
                                            if sow_text not in seen:
                                                results.append(sow_text)
                                                seen.add(sow_text)
                                    
                                    st.session_state['sow_search_results'][topic_id] = results
                
                st.session_state['sow_loaded'] = True
                loading_placeholder.empty()
            
            if st.session_state.get('sow_loaded', False):
                selected_topic = st.selectbox(
                    '🎯 Select a topic to view',
                    options=[
                        f"{ar_topics[topic_id]} ({prob:.1%})"
                        for topic_id, prob in topic_probs
                        if topic_id in ar_topics
                    ],
                    key='selected_topic_filter'
                )
                
                selected_text = selected_topic.rsplit(' (', 1)[0]
                filtered_topics = [
                    (topic_id, prob) for topic_id, prob in topic_probs
                    if topic_id in ar_topics and ar_topics[topic_id] == selected_text
                ]
                
                for topic_id, prob in filtered_topics:
                    if topic_id in ar_topics:
                        topic_text = ar_topics[topic_id]
                        results = st.session_state['sow_search_results'].get(topic_id, [])
                        
                        if results:
                            rows = []
                            for idx, sow in enumerate(results):
                                import hashlib
                                sow_key = hashlib.md5(f"{topic_id}_{idx}_{sow[:50]}".encode()).hexdigest()
                                
                                if sow.startswith('[') and sow.endswith(']'):
                                    import ast
                                    try:
                                        sow_list = ast.literal_eval(sow)
                                        if isinstance(sow_list, list):
                                            sow_items = sow_list
                                            sow_display = '\n'.join(f"• {item}" for item in sow_list)
                                        else:
                                            sow_items = [sow]
                                            sow_display = sow
                                    except:
                                        sow_items = [sow]
                                        sow_display = sow
                                else:
                                    sow_items = [sow]
                                    sow_display = sow
                                
                                is_selected = sow_key in st.session_state.get('temp_sow_selections', {})
                                
                                rows.append({
                                    'topic_id': topic_id,
                                    'sow_key': sow_key,
                                    'AR Topic': topic_text,
                                    'Probability': f"{prob:.1%}",
                                    'Statement of Work': sow_display,
                                    'sow_items': sow_items,
                                    'Selected': is_selected
                                })
                            
                            df = pd.DataFrame(rows)
                            
                            sow_per_page = 5
                            total_sows = len(df)
                            
                            sow_page_key = f'sow_page_{topic_id}'
                            if sow_page_key not in st.session_state:
                                st.session_state[sow_page_key] = 0
                            
                            total_sow_pages = (total_sows + sow_per_page - 1) // sow_per_page
                            
                            sow_start_idx = st.session_state[sow_page_key] * sow_per_page
                            sow_end_idx = min(sow_start_idx + sow_per_page, total_sows)
                            
                            filter_suffix = selected_topic.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')[:50]
                            editor_key = f"sow_editor_{topic_id}_page_{st.session_state[sow_page_key]}_{filter_suffix}"
                            
                            for i in range(sow_start_idx, sow_end_idx):
                                row = df.iloc[i]
                                col1, col2 = st.columns([0.9, 0.1])
                                with col1:
                                    sow_text = row['Statement of Work']
                                    st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>{sow_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                with col2:
                                    checkbox_key = f"{editor_key}_{i}"
                                    is_selected = st.checkbox("", value=row['Selected'], key=checkbox_key, label_visibility="collapsed")
                                    
                                    sow_key = row['sow_key']
                                    if is_selected:
                                        st.session_state['temp_sow_selections'][sow_key] = {
                                            'ar_topic': row['AR Topic'],
                                            'probability': row['Probability'],
                                            'sow_items': row['sow_items']
                                        }
                                    else:
                                        if sow_key in st.session_state['temp_sow_selections']:
                                            del st.session_state['temp_sow_selections'][sow_key]
                            
                            if total_sows > sow_per_page:
                                st.divider()
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col1:
                                    if st.button('⬅️ Previous', disabled=(st.session_state[sow_page_key] == 0), key=f'prev_sow_{topic_id}'):
                                        st.session_state[sow_page_key] -= 1
                                        st.rerun()
                                with col2:
                                    st.markdown(f"<div style='text-align: center'>**SOW {st.session_state[sow_page_key] + 1} of {total_sow_pages}**</div>", unsafe_allow_html=True)
                                with col3:
                                    if st.button('Next ➡️', disabled=(st.session_state[sow_page_key] >= total_sow_pages - 1), key=f'next_sow_{topic_id}'):
                                        st.session_state[sow_page_key] += 1
                                        st.rerun()
                        else:
                            st.info("No matching SOW found")
                
                st.divider()
                num_selected = len(st.session_state.get('temp_sow_selections', {}))
                st.info(f"📊 {num_selected} SOW(s) selected")
                
                st.info("💡 Make your selections above, then click the button below once when you're done.")
                if st.button("✅ Confirm All SOW Selections", type="primary", use_container_width=True):
                    st.session_state['selected_sows'] = st.session_state['temp_sow_selections'].copy()
                    st.session_state['sow_confirmed'] = True
                    st.rerun()
        else:
            st.info(f"No topics found with probability >= {min_probability:.0%}")
    
    except Exception as e:
        st.error(f"Error predicting AR topics: {e}")


def _search_documents(query: str, index: Dict, topics: Dict, max_results: int = 10) -> List[str]:
    """Search documents using inverted index.
    
    Args:
        query: Search query
        index: Inverted index from build_index
        topics: Topics dictionary from build_index
        max_results: Maximum results to return
        
    Returns:
        List of matching topic strings
    """
    from cap_search import _tokenize_and_stem
    
    query_tokens = _tokenize_and_stem(query)
    
    if not query_tokens:
        return []
    
    accumulator: Dict[str, int] = {}
    
    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                accumulator[doc_id] = accumulator.get(doc_id, 0) + 1
    
    if not accumulator:
        return []
    
    sorted_docs = sorted(
        accumulator.items(),
        key=lambda item: item[1],
        reverse=True
    )[:max_results]
    
    return [topics[doc_id] for doc_id, _ in sorted_docs if doc_id in topics]


def render_recommendation_mode(esf_model: Optional[object], ar_topic_model: Optional[object]) -> None:
    """Render mission assignment recommendation interface.
    
    Args:
        esf_model: ESF prediction model
        ar_topic_model: AR topic prediction model
    """
    st.header('Step 1: Input Incident Information')
    
    if 'reset_counter' not in st.session_state:
        st.session_state['reset_counter'] = 0
    
    incident_types = st.multiselect(
        'Select incident type(s)',
        INCIDENT_TYPES,
        help='Choose one or more incident types',
        key=f'incident_types_{st.session_state["reset_counter"]}'
    )
    
    declaration_type = st.selectbox(
        'Select declaration type',
        DECLARATION_TYPES,
        help='DR = Disaster Declaration, EM = Emergency Declaration',
        key=f'declaration_type_{st.session_state["reset_counter"]}'
    )
    
    state = st.selectbox(
        'Select state',
        STATES,
        help='Choose the affected state or territory',
        key=f'state_{st.session_state["reset_counter"]}'
    )
    
    current_inputs = {
        'incident_types': sorted(incident_types),
        'declaration_type': declaration_type,
        'state': state
    }
    
    inputs_changed = False
    if 'last_esf_inputs' in st.session_state:
        if st.session_state['last_esf_inputs'] != current_inputs:
            inputs_changed = True
    
    if not incident_types:
        st.warning(ERROR_MESSAGES['no_incident'])
        return
    
    if inputs_changed and 'esf_categories' in st.session_state:
        st.warning('⚠️ Input values have changed. Click "Generate ESF Prediction" to update predictions.')
    
    if st.button('Generate ESF Prediction', type='primary'):
        if esf_model is None:
            st.error("ESF model not loaded. Cannot generate predictions.")
            return
        
        with st.spinner('Generating ESF predictions...'):
            state_abbrev = STATE_ABBREV.get(state, state)
            input_features = stage_one_input(incident_types, state_abbrev, declaration_type)
            predictions = esf_model.predict_proba(input_features)
            categories = categorize_probabilities(predictions)
            
            st.session_state['esf_categories'] = categories
            st.session_state['stage1_features'] = input_features
            st.session_state['last_esf_inputs'] = current_inputs
            
            if 'selected_esfs' in st.session_state:
                st.session_state['selected_esfs'] = []
            if 'show_ar_topics' in st.session_state:
                del st.session_state['show_ar_topics']
            if 'ar_topic_probs' in st.session_state:
                del st.session_state['ar_topic_probs']
            if 'selected_sows' in st.session_state:
                st.session_state['selected_sows'] = {}
    
    if 'esf_categories' in st.session_state:
        st.divider()
        st.subheader('ESF Predictions')
        display_esf_predictions(st.session_state['esf_categories'])
    
    st.divider()
    st.header('Step 2: Review Selected ESFs')
    
    esf_selection = st.session_state.get('selected_esfs', [])
    
    esf_selection_changed = False
    if 'last_esf_selection' in st.session_state:
        if sorted(st.session_state['last_esf_selection']) != sorted(esf_selection):
            esf_selection_changed = True
    
    if esf_selection:
        with st.container():
            st.markdown(f"**{len(esf_selection)} ESF(s) Selected:**")
            cols = st.columns(min(len(esf_selection), 4))
            for idx, esf in enumerate(esf_selection):
                with cols[idx % 4]:
                    st.success(esf)
        
        if esf_selection_changed and 'show_ar_topics' in st.session_state:
            st.warning('⚠️ ESF selection has changed. Click "Suggest Assistance Request Topics" to update recommendations.')
        
        if st.button('Suggest Assistance Request Topics', type='primary'):
            if ar_topic_model is None:
                st.error("AR topic model not loaded. Cannot generate suggestions.")
                return
            
            if 'stage1_features' not in st.session_state:
                st.error("Please generate ESF predictions first.")
                return
            
            with st.spinner('Generating topic suggestions...'):
                esf_features = pd.DataFrame(0, index=[0], columns=ESF_COLUMNS)
                for esf in esf_selection:
                    esf_features[esf] = 1
                
                combined_features = st.session_state['stage1_features'].join(esf_features)
                predictions = ar_topic_model.predict_proba(combined_features)
                
                topic_probs = [
                    (idx, predictions[0, idx])
                    for idx in range(predictions.shape[1])
                    if predictions[0, idx] >= 0.10
                ]
                
                topic_probs.sort(key=lambda x: x[1], reverse=True)
                
                st.session_state['ar_topic_probs'] = topic_probs
                st.session_state['show_ar_topics'] = True
                st.session_state['last_esf_selection'] = esf_selection.copy()
                st.session_state['ar_topics_page'] = 0
                
                for key in ['selected_sows', 'temp_sow_selections', 'sow_search_results', 'sow_loaded', 'sow_confirmed']:
                    if key in st.session_state:
                        del st.session_state[key]
        
        if st.session_state.get('show_ar_topics', False):
            st.divider()
            st.subheader('Suggested Assistance Request Topics')
            display_ar_topics()
    else:
        st.info("Select ESFs from the predictions above using the checkboxes.")
    
    st.divider()
    if st.session_state.get('selected_sows', {}):
        st.header('📥 Export Selected Statements of Work')
        
        num_selected = len(st.session_state['selected_sows'])
        st.info(f"You have selected {num_selected} Statement(s) of Work")
        
        csv_data = []
        for sow_key, sow_data in st.session_state['selected_sows'].items():
            for item in sow_data['sow_items']:
                csv_data.append({
                    'AR Topic': sow_data['ar_topic'],
                    'Probability': sow_data['probability'],
                    'Statement of Work': item
                })
        
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📄 Build and Download CSV",
            data=csv,
            file_name='selected_statements_of_work.csv',
            mime='text/csv',
            type='primary',
            use_container_width=True
        )
    else:
        st.info("Select Statements of Work from the suggestions above to enable export.")
    
    st.divider()
    if st.button('🔄 Reset and Start Over', use_container_width=True, type='secondary'):
        st.session_state['reset_counter'] = st.session_state.get('reset_counter', 0) + 1
        
        all_keys = list(st.session_state.keys())
        for key in all_keys:
            if key not in ['previous_mode', 'reset_counter']:
                del st.session_state[key]
        
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


def render_capability_search_mode() -> None:
    """Render capability-based search interface."""
    st.header('Capability-Based Search')
    
    if 'capability_reset_counter' not in st.session_state:
        st.session_state['capability_reset_counter'] = 0
    if 'capability_sow_selections' not in st.session_state:
        st.session_state['capability_sow_selections'] = {}
    if 'capability_ar_results' not in st.session_state:
        st.session_state['capability_ar_results'] = []
    if 'capability_sow_results' not in st.session_state:
        st.session_state['capability_sow_results'] = []
    if 'capability_search_done' not in st.session_state:
        st.session_state['capability_search_done'] = False
    
    capability = st.text_input(
        'Enter a capability or keyword',
        placeholder='e.g., medical services, power restoration',
        key=f'capability_input_{st.session_state["capability_reset_counter"]}'
    )
    
    if st.button('🔍 Search', type='primary'):
        if capability and capability.strip():
            st.session_state['capability_sow_selections'] = {}
            st.session_state['capability_sow_page'] = 0
            
            with st.spinner('Searching...'):
                sow_index, sow_topics = build_index(str(SOW_TOPICS_FILE))
                if sow_index and sow_topics:
                    sow_results = _search_documents(capability, sow_index, sow_topics)
                    st.session_state['capability_sow_results'] = sow_results
                
                st.session_state['capability_search_done'] = True
                st.rerun()
        else:
            st.warning('Please enter a capability to search.')
    
    if st.session_state['capability_search_done'] and st.session_state['capability_sow_results']:
        st.divider()
        st.subheader('Found Statements of Work')
        
        sow_results = st.session_state['capability_sow_results']
        
        if sow_results:
            rows = []
            for idx, sow in enumerate(sow_results):
                import hashlib
                import ast
                import re
                
                sow_key = hashlib.md5(f"cap_{idx}_{sow[:50]}".encode()).hexdigest()
                
                cleaned_sow = re.sub(r'^\s*\d+\s*[-.]?\s*', '', sow)
                
                if cleaned_sow.startswith('[') and cleaned_sow.endswith(']'):
                    try:
                        sow_list = ast.literal_eval(cleaned_sow)
                        if isinstance(sow_list, list):
                            sow_items = sow_list
                            sow_display = '\n'.join(f"• {item}" for item in sow_list)
                        else:
                            sow_items = [cleaned_sow]
                            sow_display = cleaned_sow
                    except Exception as e:
                        sow_items = [cleaned_sow]
                        sow_display = cleaned_sow
                else:
                    sow_items = [cleaned_sow]
                    sow_display = cleaned_sow
                
                is_selected = sow_key in st.session_state.get('capability_sow_selections', {})
                
                rows.append({
                    'sow_key': sow_key,
                    'Statement of Work': sow_display,
                    'sow_items': sow_items,
                    'Selected': is_selected
                })
            
            df = pd.DataFrame(rows)
            
            sow_per_page = 5
            total_sows = len(df)
            
            if 'capability_sow_page' not in st.session_state:
                st.session_state['capability_sow_page'] = 0
            
            total_sow_pages = (total_sows + sow_per_page - 1) // sow_per_page
            
            if st.session_state['capability_sow_page'] >= total_sow_pages:
                st.session_state['capability_sow_page'] = 0
            
            sow_start_idx = st.session_state['capability_sow_page'] * sow_per_page
            sow_end_idx = min(sow_start_idx + sow_per_page, total_sows)
            
            for i in range(sow_start_idx, sow_end_idx):
                row = df.iloc[i]
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    sow_text = row['Statement of Work']
                    st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>{sow_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                with col2:
                    checkbox_key = f"cap_sow_check_{i}_{st.session_state['capability_sow_page']}"
                    checkbox_is_selected = st.checkbox("", value=row['Selected'], key=checkbox_key, label_visibility="collapsed")
                    
                    sow_key = row['sow_key']
                    if checkbox_is_selected:
                        st.session_state['capability_sow_selections'][sow_key] = {
                            'sow_items': row['sow_items']
                        }
                    else:
                        if sow_key in st.session_state['capability_sow_selections']:
                            del st.session_state['capability_sow_selections'][sow_key]
            
            if total_sows > sow_per_page:
                st.divider()
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button('⬅️ Previous', disabled=(st.session_state['capability_sow_page'] == 0), key='prev_sow_cap'):
                        st.session_state['capability_sow_page'] -= 1
                        st.rerun()
                with col2:
                    st.markdown(f"<div style='text-align: center'>**SOW {st.session_state['capability_sow_page'] + 1} of {total_sow_pages}**</div>", unsafe_allow_html=True)
                with col3:
                    if st.button('Next ➡️', disabled=(st.session_state['capability_sow_page'] >= total_sow_pages - 1), key='next_sow_cap'):
                        st.session_state['capability_sow_page'] += 1
                        st.rerun()
    
    if st.session_state.get('capability_sow_selections', {}):
        st.divider()
        st.header('📥 Export Selected Statements of Work')
        
        num_selected = len(st.session_state['capability_sow_selections'])
        st.info(f"You have selected {num_selected} Statement(s) of Work")
        
        csv_data = []
        for sow_key, sow_data in st.session_state['capability_sow_selections'].items():
            for item in sow_data['sow_items']:
                csv_data.append({
                    'Statement of Work': item
                })
        
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📄 Build and Download CSV",
            data=csv,
            file_name='capability_search_statements_of_work.csv',
            mime='text/csv',
            type='primary',
            use_container_width=True
        )
        
        if st.button('🔄 Reset and Start Over', use_container_width=True):
            st.session_state['capability_reset_counter'] = st.session_state.get('capability_reset_counter', 0) + 1
            for key in ['capability_sow_selections', 'capability_sow_results', 'capability_search_done', 'capability_sow_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(f"🚨 {APP_TITLE}")
    st.markdown(
        "AI-powered mission assignment recommendations for emergency response coordination."
    )
    
    esf_model, ar_topic_model = load_models()
    
    st.divider()
    
    if 'previous_mode' not in st.session_state:
        st.session_state['previous_mode'] = None
    
    mode = st.selectbox(
        'Select an option:',
        [
            'Recommend mission assignments',
            'Select mission assignments based on capability'
        ],
        index=0
    )
    
    if st.session_state['previous_mode'] is not None and st.session_state['previous_mode'] != mode:
        keys_to_keep = ['previous_mode']
        keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]
    
    st.session_state['previous_mode'] = mode
    
    st.divider()
    
    if mode == 'Recommend mission assignments':
        render_recommendation_mode(esf_model, ar_topic_model)
    else:
        render_capability_search_mode()


if __name__ == '__main__':
    main()




