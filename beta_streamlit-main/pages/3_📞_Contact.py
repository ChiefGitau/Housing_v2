import streamlit as st
import folium
from streamlit_folium import st_folium
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import global translation system
from global_translations import t, display_language_selector, get_session_language_code

# Page configuration
st.set_page_config(
    page_title=f"📞 {t('contact')}",
    page_icon="📞",
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
    
    # Current page indicator
    st.markdown(f"**📞 {t('contact')}** ← {t('current_page', 'Current Page')}")
    
    # Navigation buttons
    if st.button(f"🏠 {t('home', 'Home')}", use_container_width=True):
        st.switch_page("app.py")
    
    if st.button(f"💬 {t('chat')}", use_container_width=True):
        st.switch_page("pages/1_💬_Chat.py")
    
    if st.button(f"📄 {t('document_help')}", use_container_width=True):
        st.switch_page("pages/2_📄_Document_Help.py")
    
    if st.button(f"🔒 {t('data_upload')}", use_container_width=True):
        st.switch_page("pages/4_🔒_Admin_Upload.py")

# Local services data for different languages
SERVICES_DATA = {
    'en': [
        'Housing Rights Information',
        'Rental Assistance Programs', 
        'Tenant-Landlord Mediation',
        'Fair Housing Complaints',
        'Emergency Housing Services'
    ],
    'es': [
        'Información sobre Derechos de Vivienda',
        'Programas de Asistencia de Alquiler',
        'Mediación Inquilino-Propietario',
        'Quejas de Vivienda Justa',
        'Servicios de Vivienda de Emergencia'
    ],
    'fr': [
        'Informations sur les Droits au Logement',
        'Programmes d\'Assistance Locative',
        'Médiation Locataire-Propriétaire',
        'Plaintes de Logement Équitable',
        'Services de Logement d\'Urgence'
    ],
    'de': [
        'Informationen zu Wohnrechten',
        'Mietbeihilfeprogramme',
        'Mieter-Vermieter-Mediation',
        'Beschwerden über faires Wohnen',
        'Notunterkunftsdienste'
    ],
    'nl': [
        'Informatie over Woonrechten',
        'Huurondersteuningsprogramma\'s',
        'Huurder-Verhuurder Bemiddeling',
        'Klachten over Eerlijke Huisvesting',
        'Noodhuisvestingsdiensten'
    ]
}

def get_services_for_language(language='en'):
    """Get services list for the current language."""
    return SERVICES_DATA.get(language, SERVICES_DATA['en'])

def get_office_data(city='amsterdam'):
    """Get office data based on city selection."""
    office_data = {
        'amsterdam': {
            'lat': 52.3676,
            'lng': 4.9041,
            'address': 'Nieuwezijds Voorburgwal 147\n1012 RJ Amsterdam\nNetherlands',
            'phone': '+31 20 555 0123',
            'hours': 'Monday - Friday: 9:00 AM - 5:00 PM\nSaturday: 10:00 AM - 2:00 PM\nSunday: Closed'
        },
        'leeuwarden': {
            'lat': 53.2012,
            'lng': 5.8086,
            'address': 'Wilhelminaplein 1\n8911 BS Leeuwarden\nNetherlands',
            'phone': '+31 58 555 0456',
            'hours': 'Monday - Friday: 8:30 AM - 4:30 PM\nSaturday: 9:00 AM - 1:00 PM\nSunday: Closed'
        }
    }
    return office_data.get(city.lower(), office_data['amsterdam'])

def create_map(city='amsterdam'):
    """Create a map with dummy government office location."""
    office_data = get_office_data(city)
    office_lat = office_data['lat']
    office_lng = office_data['lng']
    
    # Create map centered on the office location
    m = folium.Map(
        location=[office_lat, office_lng],
        zoom_start=13,
        width='100%',
        height='400px'
    )
    
    # Add marker for government office
    folium.Marker(
        [office_lat, office_lng],
        popup="Government Housing Office",
        tooltip="Government Housing Office",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)
    
    return m 

# Get user's language
user_language = get_session_language_code()

# Header
st.header(t('contact_help'))
st.subheader(t('find_local_assistance'))

if user_language != 'en':
    st.info(f"Content displayed in: {user_language.upper()}")

# Location selector
st.markdown(f"### {t('Search by Location')}")
location_input = st.selectbox(
    f"{t('Select a city to find the nearest government housing office:')}",
    options=["Amsterdam", "Leeuwarden"],
    index=0
)

selected_city = location_input.lower()

# Create two columns: map on left, details on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{t('Nearest Government Housing Office')}")
    
    # Create and display map
    office_map = create_map(selected_city)
    map_data = st_folium(office_map, width=700, height=400)

with col2:
    st.subheader(t('office_details'))
    
    # Get office data for selected city
    office_data = get_office_data(selected_city)
    
    # Display office contact information
    st.markdown(f"**{t('address')}:**")
    st.text(office_data['address'])
    
    st.markdown(f"**{t('phone')}:**")
    st.text(office_data['phone'])
    
    st.markdown(f"**{t('hours')}:**")
    st.text(office_data['hours'])

# Services section
st.divider()
st.subheader(t('services_available'))

services = get_services_for_language(user_language)
for service in services:
    st.write(f"• {service}")

# Emergency and important notes
st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader(t('emergency_housing'))
    st.warning(t("For immediate housing emergencies, contact your local emergency services."))

with col4:
    st.subheader(t('important_note'))
    st.info(t("This is sample location data for demonstration purposes. Contact information shown is not real."))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    LAISA Contact Information - For demonstration purposes only
</div>
""", unsafe_allow_html=True)