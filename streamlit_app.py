import streamlit as st

st.set_page_config(layout="wide")

st.title('Time Series Analysis with Statistical Significance, Clustering, ARIMA')

with st.expander('About this app'):
  app_description = """This app contains analysis process for my final college project (Skripsi) focusing on time series analysis using an ARIMA model, with data preprocessing techniques, including statistical feature selection and cluster-based outlier detection. The primary goal of this project is to build a robust model for a given time series dataset (e.g., PM2.5 air quality)."""
    # 
  st.write(app_description)
  st.link_button("Check My GitHub", "https://github.com/MuhammadKurniaSani-me/final-college-project")

if st.checkbox("Enable CSS hacks", True):
    
    titleFontSize = "40px"
    titleFontWeight = "500"
    headerFontSize = "32px"
    headerFontWeight = "500"
    subheaderFontSize = "24px"
    subheaderFontWeight = "500"
    
    pageHoverBackgroundColor = "#deddd1"
    pageFontSize = "14px"
    
    activePageBackgroundColor = "#deddd1"
    activePageHoverBackgroundColor = "#deddd1"
    
    
    st.html(
        f"""
        <style>
        body {{
            -webkit-font-smoothing: antialiased;
        }}
        
        h1 {{
            font-size: {titleFontSize} !important;
            font-weight: {titleFontWeight} !important;
        }}
        
        h2 {{
            font-size: {headerFontSize} !important;
            font-weight: {headerFontWeight} !important;
        }}
        
        h3 {{
            font-size: {subheaderFontSize} !important;
            font-weight: {subheaderFontWeight} !important;
        }}
        
        /* Active page in sidebar nav */
        [data-testid="stSidebarNav"] li a[aria-current="page"] {{
            background-color: {activePageBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li a[aria-current="page"]:hover {{
            background-color: {activePageHoverBackgroundColor} !important;
        }}
        
        /* Other pages in sidebar nav */
        [data-testid="stSidebarNav"] li a:hover {{
            background-color: {pageHoverBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li a span {{
            font-size: {pageFontSize} !important;
        }}
        </style>
        """
    )

st.markdown("   ")

pg = st.navigation(
    {
        "General": [
            st.Page("./pages/1_introduction.py", title="Introduction", icon=":material/home:"),
            st.Page("./pages/2_data_overview.py", title="Data Overview", icon=":material/table_chart:"),
            st.Page("./pages/3_data_preprocessing.py", title="Preprocessing", icon=":material/tune:"),
            st.Page("./pages/4_statistical_significance.py", title="Statistical Significance", icon=":material/functions:"),
            st.Page("./pages/5_outlier_removal.py", title="Outlier Removal", icon=":material/filter_alt_off:"),
            st.Page("./pages/6_arima.py", title="ARIMA", icon=":material/timeline:"),
            st.Page("./pages/7_conlusion.py", title="Conclusion", icon=":material/flag:"),
        ],
        # "Admin": [st.Page(page3, title="Settings", icon=":material/settings:")],
    }
)


pg.run()



# # This should display a header titled "Input"
# st.header('Input') 
    
# # These should display input widgets
# user_name = st.text_input('What is your name?')
# user_emoji = st.selectbox('Choose an emoji', ['', '😄', '😆', '😊', '😍', '😴', '😕', '😱'])
# user_food = st.selectbox('What is your favorite food?', ['', 'Tom Yum Kung', 'Burrito', 'Lasagna', 'Hamburger', 'Pizza'])

# # This should display a header titled "Output"
# st.header('Output')

# # This sets up three columns for layout
# col1, col2, col3 = st.columns(3)

# # The 'with' blocks should display content within each column.
# # Initially, they will show the "Please enter/choose" messages.
# with col1:
#     if user_name != '':
#         st.write(f'👋 Hello {user_name}!')
#     else:
#         st.write('👈  Please enter your **name**!')

# with col2:
#     if user_emoji != '':
#         st.write(f'{user_emoji} is your favorite **emoji**!')
#     else:
#         st.write('👈 Please choose an **emoji**!')

# with col3:
#     if user_food != '':
#         st.write(f'🍴 **{user_food}** is your favorite **food**!')
#     else:
#         st.write('👈 Please choose your favorite **food**!')

# st.sidebar.header('Input')
# user_name = st.sidebar.text_input('What is your name?')
# user_emoji = st.sidebar.selectbox('Choose an emoji', ['', '😄', '😆', '😊', '😍', '😴', '😕', '😱'])
# user_food = st.sidebar.selectbox('What is your favorite food?', ['', 'Tom Yum Kung', 'Burrito', 'Lasagna', 'Hamburger', 'Pizza'])

# st.header('Output')

# col1, col2, col3 = st.columns(3)

# with col1:
#   if user_name != '':
#     st.write(f'👋 Hello {user_name}!')
#   else:
#     st.write('👈  Please enter your **name**!')

# with col2:
#   if user_emoji != '':
#     st.write(f'{user_emoji} is your favorite **emoji**!')
#   else:
#     st.write('👈 Please choose an **emoji**!')

# with col3:
#   if user_food != '':
#     st.write(f'🍴 **{user_food}** is your favorite **food**!')
#   else:
#     st.write('👈 Please choose your favorite **food**!')


# my_bar = st.progress(0)

# for percent_complete in range(100):
#      time.sleep(0.05)
#      my_bar.progress(percent_complete + 1)

# st.balloons()


# # st.button
# # st.header('st.button')

# # if st.button('Say hello'):
# #     st.write('Why hello there')
# # else:
# #     st.write('Goodbye')


# st.header('st.write')

# # Example 1

# st.write('Hello, *World!* :sunglasses:')

# # Example 2

# st.write(1234)

# # Example 3

# df = pd.DataFrame({
#      'first column': [1, 2, 3, 4],
#      'second column': [10, 20, 30, 40]
#      })
# st.write(df)

# # Example 4

# st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# # Example 5

# df2 = pd.DataFrame(
#      np.random.randn(200, 3),
#      columns=['a', 'b', 'c'])
# c = alt.Chart(df2).mark_circle().encode(
#      x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
# st.write(c)


# st.header('st.slider')

# # Example 1

# st.subheader('Slider')

# age = st.slider('How old are you?', 0, 130, 25)
# st.write("I'm ", age, 'years old')

# # Example 2

# st.subheader('Range slider')

# values = st.slider(
#      'Select a range of values',
#      0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)

# # Example 3

# st.subheader('Range time slider')

# appointment = st.slider(
#      "Schedule your appointment:",
#      value=(time.time(11, 30), time.time(12, 45)))
# st.write("You're scheduled for:", appointment)

# # Example 4

# st.subheader('Datetime slider')

# start_time = st.slider(
#      "When do you start?",
#      value=datetime(2020, 1, 1, 9, 30),
#      format="MM/DD/YY - hh:mm")
# st.write("Start time:", start_time)


# st.header('Line chart')

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)


# st.header('st.selectbox')

# option = st.selectbox(
#      'What is your favorite color?',
#      ('Blue', 'Red', 'Green'))

# st.write('Your favorite color is ', option)





# st.header('st.checkbox')

# st.write ('What would you like to order?')

# icecream = st.checkbox('Ice cream')
# coffee = st.checkbox('Coffee')
# cola = st.checkbox('Cola')

# if icecream:
#      st.write("Great! Here's some more 🍦")

# if coffee: 
#      st.write("Okay, here's some coffee ☕")

# if cola:
#      st.write("Here you go 🥤")


# st.header('st.latex')

# st.latex(r'''
#      a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#      \sum_{k=0}^{n-1} ar^k =
#      a \left(\frac{1-r^{n}}{1-r}\right)
#      ''')


# st.title('Customizing the theme of Streamlit apps')

# st.write('Contents of the `.streamlit/config.toml` file of this app')

# st.code("""
# [theme]
# primaryColor="#F39C12"
# backgroundColor="#2E86C1"
# secondaryBackgroundColor="#AED6F1"
# textColor="#FFFFFF"
# font="monospace"
# """)

# number = st.sidebar.slider('Select a number:', 0, 10, 5)
# st.write('Selected number from slider widget is:', number)


# st.title('st.file_uploader')

# st.subheader('Input CSV')
# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   st.subheader('DataFrame')
#   st.write(df)
#   st.subheader('Descriptive Statistics')
#   st.write(df.describe())
# else:
#   st.info('☝️ Upload a CSV file')



# st.title('st.form')

# # Full example of using the with notation
# st.header('1. Example of using `with` notation')
# st.subheader('Coffee machine')

# with st.form('my_form'):
#     st.subheader('**Order your coffee**')

#     # Input widgets
#     coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
#     coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
#     brewing_val = st.selectbox('Brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
#     serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
#     milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
#     owncup_val = st.checkbox('Bring own cup')

#     # Every form must have a submit button
#     submitted = st.form_submit_button('Submit')

# if submitted:
#     st.markdown(f'''
#         ☕ You have ordered:
#         - Coffee bean: `{coffee_bean_val}`
#         - Coffee roast: `{coffee_roast_val}`
#         - Brewing: `{brewing_val}`
#         - Serving type: `{serving_type_val}`
#         - Milk: `{milk_val}`
#         - Bring own cup: `{owncup_val}`
#         ''')
# else:
#     st.write('☝️ Place your order!')


# # Short example of using an object notation
# st.header('2. Example of object notation')

# form = st.form('my_form_2')
# selected_val = form.slider('Select a value')
# form.form_submit_button('Submit')

# st.write('Selected value: ', selected_val)