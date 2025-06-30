import streamlit as st
import subprocess
import sys
import os

# Import global translation system
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from global_translations import t, display_language_selector, set_language_preference

# Railway port configuration
PORT = int(os.environ.get("PORT", 8501))

st.set_page_config(
    page_title=f"LAISA - {t('housing_assistant')}",
    page_icon=":robot_face:",
    layout="wide"
)

# Custom CSS to hide default navigation
st.markdown("""
<style>
    .css-1d391kg {display: none}
    .css-1rs6os {display: none}
    .css-17ziqus {display: none}
    [data-testid="stSidebarNav"] {display: none}
    .css-1544g2n {display: none}
</style>
""", unsafe_allow_html=True)

# Display language selector in sidebar
with st.sidebar:
    display_language_selector()
    
    # Custom navigation menu
    st.markdown("---")
    st.markdown(f"### {t('navigation')}")
    
    # Navigation buttons
    if st.button(f"💬 {t('chat')}", use_container_width=True):
        st.switch_page("pages/1_💬_Chat.py")
    
    if st.button(f"📄 {t('document_help')}", use_container_width=True):
        st.switch_page("pages/2_📄_Document_Help.py")
    
    if st.button(f"📞 {t('contact')}", use_container_width=True):
        st.switch_page("pages/3_📞_Contact.py")
    
    if st.button(f"🔒 {t('data_upload')}", use_container_width=True):
        st.switch_page("pages/4_🔒_Admin_Upload.py")

# Run setup check on first load
def check_setup():
    """Run setup test and return results."""
    try:
        # Run setup_test.py and capture output
        result = subprocess.run(
            [sys.executable, "setup_test.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Setup test timed out"
    except Exception as e:
        return False, "", str(e)

# Check if setup test should be run
if 'setup_checked' not in st.session_state:
    st.session_state.setup_checked = False

if not st.session_state.setup_checked:
    with st.spinner(f"{t('loading')}..."):
        success, stdout, stderr = check_setup()
        st.session_state.setup_checked = True
        st.session_state.setup_success = success
        st.session_state.setup_output = stdout
        st.session_state.setup_error = stderr

# Show setup results
if not st.session_state.setup_success:
    st.error(f" {t('system_check_failed')}")
    st.markdown("Please fix the following issues before using LAISA:")
    
    if st.session_state.setup_error:
        st.code(st.session_state.setup_error, language="text")
    
    if st.session_state.setup_output:
        with st.expander("Detailed Setup Output"):
            st.code(st.session_state.setup_output, language="text")

    
    if st.button(f"{t('check_system_status')}"):
        st.session_state.setup_checked = False
        st.rerun()
    
    st.stop()
else:
    st.success(f" {t('all_systems_operational')}! LAISA is ready to use.")
    
    # Welcome buttons in different languages
    st.markdown("### Welcome / Bienvenido / Bienvenue / Willkommen / Welkom / Benvenuto / Bem-vindo / Καλώς ήρθατε / Добро пожаловать / ようこそ")
    
    # First row of language buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🇺🇸 Welcome", help="English"):
            set_language_preference("en")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("🇪🇸 Bienvenido", help="Spanish"):
            set_language_preference("es")
            st.balloons()
            st.rerun()
    
    with col3:
        if st.button("🇫🇷 Bienvenue", help="French"):
            set_language_preference("fr")
            st.balloons()
            st.rerun()
    
    with col4:
        if st.button("🇩🇪 Willkommen", help="German"):
            set_language_preference("de")
            st.balloons()
            st.rerun()
    
    with col5:
        if st.button("🇳🇱 Welkom", help="Dutch"):
            set_language_preference("nl")
            st.balloons()
            st.rerun()
    
    # Second row of language buttons
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        if st.button("🇮🇹 Benvenuto", help="Italian"):
            set_language_preference("it")
            st.balloons()
            st.rerun()
    
    with col7:
        if st.button("🇵🇹 Bem-vindo", help="Portuguese"):
            set_language_preference("pt")
            st.balloons()
            st.rerun()
    
    with col8:
        if st.button("🇬🇷 Καλώς ήρθατε", help="Greek"):
            set_language_preference("el")
            st.balloons()
            st.rerun()
    
    with col9:
        if st.button("🇷🇺 Добро пожаловать", help="Russian"):
            set_language_preference("ru")
            st.balloons()
            st.rerun()
    
    with col10:
        if st.button("🇯🇵 ようこそ", help="Japanese"):
            set_language_preference("ja")
            st.balloons()
            st.rerun()
    
    # Third row of language buttons
    col11, col12, col13, col14, col15 = st.columns(5)
    
    with col11:
        if st.button("🇨🇳 欢迎", help="Chinese"):
            set_language_preference("zh")
            st.balloons()
            st.rerun()
    
    with col12:
        if st.button("🇰🇷 환영합니다", help="Korean"):
            set_language_preference("ko")
            st.balloons()
            st.rerun()
    
    with col13:
        if st.button("🇦🇪 أهلا وسهلا", help="Arabic"):
            set_language_preference("ar")
            st.balloons()
            st.rerun()
    
    with col14:
        if st.button("🇮🇳 स्वागत है", help="Hindi"):
            set_language_preference("hi")
            st.balloons()
            st.rerun()
    
    with col15:
        if st.button("🇹🇷 Hoş geldiniz", help="Turkish"):
            set_language_preference("tr")
            st.balloons()
            st.rerun()

st.write(f"# LAISA - {t('housing_assistant')}! :robot_face:")

st.sidebar.success(t('navigation'))

st.markdown(f"""
   {t( 'Welcome to')} LAISA {t('(Legal AI Support Assistant), your housing law and regulation assistant.')}
    
   '### '{t('Features:')}
    - {t('Ask questions about housing rights and responsibilities')}
    - {t('Get information about housing regulations')}
    - {t('Receive guidance on housing-related legal matters')}
    
    ### {t('How to use:')}
    1. {t('Navigate to the **Chat** page to start a conversation')}
    2. {t(' Ask your housing-related questions')}
    3.{t(' Review the provided information and sources')}
    '
    **{t('important_note')}**: {t('This assistant provides informational guidance only and should not be considered as legal advice.')}
    """)