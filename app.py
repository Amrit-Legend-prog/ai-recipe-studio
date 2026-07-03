from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import streamlit as st
import time, os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Recipe Studio", page_icon="🍽️", layout="centered")

st.markdown("""
<style>
body { background-color:#0e1117; }
.stButton button{border-radius:10px;}
</style>
""", unsafe_allow_html=True)

class RecipeSchema(BaseModel):
    recipe_name:str=Field(description="Recipe name")
    ingredients:list[str]
    steps:list[str]
    cooking_time:str
    difficulty_level:str
    nutrition_facts:dict[str,str]

def get_model(name):
    if name.startswith("Gemini"):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in .env")
        return ChatGoogleGenerativeAI(model="gemini-2.5-pro").with_structured_output(RecipeSchema)
    else:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in .env")
        return ChatGroq(model="llama-3.1-8b-instant").with_structured_output(RecipeSchema)

st.title("🍳 AI Recipe Studio")
st.caption("Generate structured recipes with AI")

prompt=st.text_input("Dish name or ingredients")
model=st.selectbox("Model",["Gemini (Fast)","Groq (Structured)"])
c1,c2=st.columns(2)
gen=c1.button("Generate Recipe 🍲")
cmp=c2.button("Compare Models ⚡")

def render(resp):
    st.success("Recipe Ready!")
    st.subheader(resp.recipe_name)
    st.markdown("### Ingredients")
    for i in resp.ingredients: st.write("•",i)
    st.markdown("### Steps")
    for n,s in enumerate(resp.steps,1): st.write(f"{n}. {s}")
    st.write(f"⏱ {resp.cooking_time}")
    st.write(f"🔥 {resp.difficulty_level}")
    if resp.nutrition_facts: st.json(resp.nutrition_facts)

if gen:
    if not prompt.strip():
        st.warning("Please enter a recipe prompt.")
    else:
        try:
            chain=get_model(model)
            with st.spinner("Cooking..."):
                resp=chain.invoke([SystemMessage(content="Return valid structured recipe JSON."),HumanMessage(content=prompt)])
            render(resp)
        except Exception as e:
            st.error(str(e))

if cmp:
    if not prompt.strip():
        st.warning("Please enter a recipe prompt.")
    else:
        st.subheader("Model Comparison")
        for name in ["Gemini (Fast)","Groq (Structured)"]:
            try:
                chain=get_model(name)
                with st.spinner(f"Running {name}..."):
                    t=time.time()
                    out=chain.invoke([SystemMessage(content="Return valid structured recipe JSON."),HumanMessage(content=prompt)])
                    elapsed=time.time()-t
                with st.expander(f"{name} ({elapsed:.2f}s)",expanded=True):
                    st.json(out.model_dump())
            except Exception as e:
                st.error(f"{name}: {e}")
