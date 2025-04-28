import streamlit as st
import json



with open('aorta_deepseek.json', 'r') as f:
    data = json.load(f)


def search_closest_value(data, value_mm):
    value = value_mm / 10

    if not data:
        return value
    
    closest = None
    min_diff = float('inf')
    for item in data:
        diff = abs(float(item) - value)
        if diff < min_diff:
            min_diff = diff
            closest = item
    return closest


st.set_page_config(layout="wide")
st.title("Диагностирование аорты Demo")
st.markdown("""
В демо представлены заранее сгенерированные заключения для некоторых параметров аорты:
- 1.4-6.0 см с шагом 0.3 см
- 6.0-10.4 см с шагом 0.8 см

Никакой постобработки не было, это "сырой" выход из модели - интеллектуального ассистента. \\
На данный момент есть баг с заключением при слишком маленьком диаметре аорты - пишет, что в норме.
В остальном работает отлично, как мне показалось.
            
Контакт для связи:  t.me/ktann , Олег
""".strip())

st.header("Выберите параметры аорты")


AORTA_TYPES = [
    'корня',
    'восходящей',
    'дуги',
    'нисходящей',
    'брюшной',
]
selection = st.segmented_control(
    "Часть аорты", AORTA_TYPES, selection_mode="single", default='корня'
)


slider_value = st.slider("Диаметр аорты в мм", 14, 104, 25)
closest_value = search_closest_value(data.get(selection, None), slider_value)

display_text = f"""
диаметр {selection or ''} аорты {closest_value} см

----------------------------------------

{data.get(selection, {}).get(closest_value, {})}
""".strip()

# st.text_area("Выбранные параметры и заключение", 
#              display_text,
#              height=800,
#              key="display_area")

st.markdown(display_text, unsafe_allow_html=True)