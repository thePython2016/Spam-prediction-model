import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
import streamlit as st

model=pkl.load(open("Model.pkl","rb"))
# Normalization=pkl.load(open("normalization.pkl","rb"))


st.title("Spam Prediction Model")
tab1,tab2=st.tabs(['Spam Prediction with Form',"Spam Prediction with File Upload"])


with tab1:
    # Success placeholder at top of form
    successPlaceholder = st.empty()

    # Number of Links
    errorNumLink = st.empty()
    numLink = st.number_input("Number of Links in the Email", min_value=0)

    # Number of Words
    errorNumWords = st.empty()
    numWords = st.number_input("Number of Words in the Email", min_value=0)

    # Has Offer
    errorHasOffer = st.empty()
    hasOffer = st.selectbox("Email Has Offer?", ["Email has Offer?", "Yes", "No"])

    # All Caps
    errorAllCaps = st.empty()
    allCaps = st.selectbox("Email has All Capital Letters?", ["Email has All Capital Letter", "Yes", "No"])

    button = st.button("Predict")

    def showError(placeholder, msg):
        placeholder.markdown(f"""
            <div style="
                background-color: #fff0f0;
                border-left: 4px solid #ff4b4b;
                border-radius: 4px;
                padding: 8px 12px;
                margin-bottom: 8px;
                color: #cc0000;
                font-size: 14px;
            ">{msg}</div>
        """, unsafe_allow_html=True)

    def clearAll():
        errorNumLink.empty()
        errorNumWords.empty()
        errorHasOffer.empty()
        errorAllCaps.empty()
        successPlaceholder.empty()

    if button:
        clearAll()

        if numLink <= 0:
            showError(errorNumLink, "Number of Links required")
        elif numWords <= 0:
            showError(errorNumWords, "Number of Words required")
        elif hasOffer == "Email has Offer?":
            showError(errorHasOffer, "Please Select Has Offer option")
        elif allCaps == "Email has All Capital Letter":
            showError(errorAllCaps, "Please Select All Caps option")
        else:
            myData = pd.DataFrame({
                "num_links": [numLink],
                "num_words": [numWords],
                "has_offer": [hasOffer],
                "all_caps":  [allCaps]
            })

            myData['has_offer'] = myData['has_offer'].map({"Yes": 1, "No": 0})
            myData['all_caps']  = myData['all_caps'].map({"Yes": 1, "No": 0})

            predict = model.predict(myData)
            proba   = model.predict_proba(myData)

            st.session_state.showResult = True
            st.session_state.predict    = int(predict[0])
            st.session_state.proba      = proba.tolist()

    # Show result with dismiss button
    if st.session_state.get('showResult', False):
        if st.session_state.predict == 1:
            with successPlaceholder.container():
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(f"""
                        <div style="
                            background-color: #fff0f0;
                            border-left: 4px solid #ff4b4b;
                            border-radius: 4px;
                            padding: 12px 16px;
                            color: #cc0000;
                            font-size: 16px;
                            font-weight: bold;
                        ">This email is Spam! (Confidence: {st.session_state.proba[0][1]:.2%})</div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key="dismissSpam"):
                        st.session_state.showResult = False
                        successPlaceholder.empty()
                        st.rerun()
        else:
            with successPlaceholder.container():
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(f"""
                        <div style="
                            background-color: #f0fff4;
                            border-left: 4px solid #28a745;
                            border-radius: 4px;
                            padding: 12px 16px;
                            color: #155724;
                            font-size: 16px;
                            font-weight: bold;
                        ">This email is Ham! (Confidence: {st.session_state.proba[0][0]:.2%})</div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key="dismissHam"):
                        st.session_state.showResult = False
                        successPlaceholder.empty()
                        st.rerun()
with tab2:
    file=st.file_uploader("Upload File",type="csv")
    buttonFile=st.button("Send to Predict")

    if buttonFile:
        if not file:
            st.error("Upload File to Predict")
        else:
            fileData=pd.read_csv(file)
            # Convert all columns to numeric, coercing bad values to NaN
            # fileData['num_links'] = pd.to_numeric(fileData['num_links'], errors='coerce')
            # fileData['num_words'] = pd.to_numeric(fileData['num_words'], errors='coerce')
            # fileData['has_offer'] = pd.to_numeric(fileData['has_offer'], errors='coerce')
            # fileData['all_caps']  = pd.to_numeric(fileData['all_caps'], errors='coerce')

# Fill or drop NaNs
            # fileData = fileData.fillna(0)  # or df.dropna()

            # Then predict
           
            predictSpam=model.predict(fileData)
            fileData['Predicted Outcome']=predictSpam
            fileData['Predicted Outcome'] = fileData['Predicted Outcome'].map({1: 'Spam', 0: 'Not Spam'})

            st.write(fileData)