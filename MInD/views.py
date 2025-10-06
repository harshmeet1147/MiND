from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect 
from django.contrib import messages
import csv
import json
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings 


def home(request):
    return HttpResponse("Welcome to Hive")


def dash(request):
    return render(request,'map.html')


def select_brand(request):
    return render(request, 'select_brand.html')


def brand_summary(request):
    
    return render(request, 'brand_summary.html')


def googlenews(request):
    return render(request, 'googlenews.html')

def consumersentiment(request):
    return render(request, 'consumersentiment.html')

def store(request):
    return render(request, 'store_review_analysis.html')

def playstore(request):
    return render(request, 'playstore_review_analysis.html')

def sms(request):
    return render(request, 'sms.html')




def get_chart_data(request):
    # csv_file_path = r"C:\Users\Swaraj\Desktop\ChrysCapital11\hive\volumetrends.csv"
    csv_file_path = r"C:\Users\Harshmeet\Desktop\mindv3\OngoingProjects\MInD\hive\volumetrends.csv"
    df = pd.read_csv(csv_file_path)
    print("CSV Columns:", df.columns.tolist())

    data = df.to_dict(orient="records")
    return JsonResponse(data, safe=False)


def chart_view(request):
    return render(request, 'chart.html')




def get_chart_data_buzz(request):
    # csv_file_path = r"C:\Users\Swaraj\Desktop\ChrysCapital\hive\dailybuzz.csv"
    csv_file_path = r"C:\Users\Harshmeet\Desktop\mindv3\OngoingProjects\MInD\dailybuzz.csv"

    df = pd.read_csv(csv_file_path)
    print("CSV Columns:", df.columns.tolist())

    data = df.to_dict(orient="records")
    return JsonResponse(data, safe=False)


def chart_view_buzz(request):
    return render(request, 'buzz.html')

def valuetouser(request):
    return render(request, 'valuetousers.html')

def scaleeng(request):
    return render(request, 'scaleeng.html')

def webtracking(request):
    return render(request, 'webtracking.html')

def webtracking2(request):
    return render(request, 'webtrack2.html')

def brand(request):
    return render(request, 'brand.html')

def brandstrength(request):
    return render(request, 'brandstrength.html')

def cat(request):
    return render(request, 'cat.html')

def brandsearch(request):
    return render(request, 'brandsearch.html')

def retailsov(request):
    return render(request, 'retailsov.html')

def estmarket(request):
    return render(request, 'estmarket.html')

def appuser(request):
    return render(request, 'appuser.html')

def appret(request):
    return render(request, 'appret.html')

def appperf(request):
    return render(request, 'appperf.html')


def brandstrengthoverviewtitle(request):
    return render(request,'brandstrengthoverviewtitle.html')

def customerAffinityoverviewtitle(request):
    return render(request,'customerAffinityoverview.html')
def EngagementRates(request):
    return render(request,'EngagementRates.html')

def Brandperformanceoverview(request):
    return render(request,'Brandperformanceoverview.html')

def ntbmove(request):
    return render(request,'ntbmove.html')

def appsummary(request):
    return render(request,'appsummary.html')
def quality(request):
    return render(request,'quality.html')

def mind(request):
    return render(request,'mind.html')

def ntbchannel(request):
    return render(request,'ntbchannel.html')

def appsum(request):
    return render(request,'appsum.html')

def rank(request):
    return render(request,'rank.html')


import pandas as pd
import numpy as np
from django.http import JsonResponse

def oneData(request):
    filePath = r"C:\Users\Administrator\Desktop\mindv3new\MiND\helium_data1.xlsx"
    # filePath = r"C:\Users\Harshmeet\Desktop\mindv3\OngoingProjects\MInD\helium_data1.xlsx"
    
    # Read Excel
    df = pd.read_excel(filePath)
    df.columns = df.columns.str.strip()  # Trim spaces in column names

    # Convert NaN values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else int(x) if x == x // 1 else float(x))
        elif df[col].dtype == object:
            df[col] = df[col].fillna("")

    # Ensure category column exists
    if 'category_dashb' in df.columns:
        categories = sorted(df['category_dashb'].dropna().unique().tolist())  # Sort categories alphabetically
    else:
        categories = []

    # Derived columns
    df['bsr_bucket'] = df['rank'].apply(lambda x: int(x // 10) * 10 if pd.notna(x) else None)
    df['price_bucket'] = df['price'].apply(lambda x: int(x // 200) * 200 if pd.notna(x) else None)

    # Convert DataFrame to JSON
    data = {
        "categories": categories,  # Ensure categories is always present
        "records": df.to_dict(orient="records")
    }
    
    return JsonResponse(data, safe=False)



def one(request):
    return render(request,'one.html')

def two(request):
    return render(request,'two.html')

def three(request):
    return render(request,'three.html')

def four(request):
    return render(request,'four.html')

def five(request):
    return render(request,'five.html')

def six(request):
    return render(request,'six.html')

def seven(request):
    return render(request,'seven.html')

def eight(request):
    return render(request,'eight.html')
def nine(request):
    return render(request,'nine.html')

def eleven(request):
    return render(request,'eleven.html')

def twelve(request):
    return render(request,'twelve.html')

def amc1(request):
    return render(request, 'amc1.html')

def a(request):
    return render(request,'a.html')

def d(request):
    return render(request,'d.html')

def i(request):
    return render(request,'I.html')
def b(request):
    return render(request,'b.html')

def c(request):
    return render(request,'c.html')
def L(request):
    return render(request,'L.html')

def j(request):
    return render(request,'j.html')
def e(request):
    return render(request,'e.html')

def qcom(request):
    return render(request,'qcom.html')

def hivear(request):
    return render(request,'hivear.html')



#all endpoint starts here

from .models import HeliumData

def heliumData(request):
    data= HeliumData.objects.all().values()
    return JsonResponse(list(data),safe=False)

def heliumData_a(request):
    queryset = HeliumData.objects.all()
    
    data = []
    for item in queryset:
        data.append({
            "Date":item.date,
            "Brand": item.brand,
            "Revenue": item.revenue or 0,
            "Units": item.sales or 0,
            "AOV": round((item.revenue / item.sales), 2) if item.revenue and item.sales else 0,
            "revenueshare": 0,  # You might need to calculate this
            "unitsoldshare": 0,  # Same here
            "category":item.category_dashb,
        })

    # Optional: calculate share % if not stored in DB
    total_revenue = sum(d["Revenue"] for d in data)
    total_units = sum(d["Units"] for d in data)

    for d in data:
        d["revenueshare"] = round((d["Revenue"] / total_revenue) * 100, 2) if total_revenue else 0
        d["unitsoldshare"] = round((d["Units"] / total_units) * 100, 2) if total_units else 0

    return JsonResponse(data, safe=False)

def heliumData_b(request):
    queryset = HeliumData.objects.all()
    
    data = []
    for item in queryset:
        data.append({
            "date":item.date,
            "brand": item.brand,
            "revenue": item.revenue or 0,
            "sales": item.sales or 0,
            "category":item.category_dashb,
        })


    return JsonResponse(data, safe=False)



import math
from typing import List

def compute_sov_index(search_volumes: List[float], positions: List[List[int]]) -> float:

    N = len(search_volumes)
    M = len(positions)

    if N == 0 or M == 0:
        raise ValueError("Search volumes and positions must be non-empty")

    numerator = 0.0
    for k in range(N):
        SV_k = search_volumes[k]
        exp_sum = 0.0
        for i in range(M):
            F_ik = positions[i][k]
            exp_sum += math.exp(-((F_ik - 1) / 12))
        avg_exp = exp_sum / M
        numerator += SV_k * avg_exp

    denominator = sum(search_volumes)

    if denominator == 0:
        return 0.0

    sov_index = numerator / denominator
    return sov_index






def get_bsr_bucket(bsr):
    bsr = int(bsr)
    bucket_size = 20
    lower = ((bsr - 1) // bucket_size) * bucket_size + 1
    upper = lower + bucket_size - 1
    return f"{lower}-{upper}"

def get_units_bucket(units):
    units = int(units)
    bucket_size = 50000
    lower = (units // bucket_size) * bucket_size
    upper = lower + bucket_size - 1
    return f"{lower}-{upper}"

def brand_sku_view(request):
    # csv_file_path = r'C:\Users\Swaraj\Desktop\OngoingProjects\MInD\static\data\brand_sku_data.csv'
    csv_file_path = r'C:\Users\Harshmeet\Desktop\mindv3\OngoingProjects\MInD\static\data\brand_sku_data.csv'

    with open(csv_file_path, newline='', encoding='ISO-8859-1') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    categories = sorted(set(row['category'] for row in data))
    selected_category = request.GET.get('category', categories[0])
    filtered_data = [row for row in data if row['category'] == selected_category]

    bucket_matrix = {}
    all_x = set()
    all_y = set()

    for row in filtered_data:
        bsr_bucket = get_bsr_bucket(row['bsr'])
        units_bucket = get_units_bucket(row['units'])
        key = f"{bsr_bucket}|||{units_bucket}"  # <- this must match JS

        all_x.add(bsr_bucket)
        all_y.add(units_bucket)

        bucket_matrix.setdefault(key, []).append({
            'asin': row['asin'],
            'revenue': int(row['revenue']),
            'units': int(row['units']),
            'brand': row['brand'],
        })

    return render(request, 'ab.html', {
        'categories': categories,
        'selected_category': selected_category,
        'buckets': json.dumps(bucket_matrix),
        'x_labels': json.dumps(sorted(all_x, key=lambda x: int(x.split('-')[0]))),
        'y_labels': json.dumps(sorted(all_y, key=lambda y: int(y.split('-')[0]))),
    })
def a1(request):
    return render(request,'a1.html')

def a2(request):
    return render(request,'a2.html')

def a3(request):
    return render(request,'a3.html')

def a4(request):
    return render(request,'a4.html')

def a5(request):
    return render(request,'a5.html')


def a6(request):
    return render(request,'a6.html')


import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


CLIENT_ID = 'hiveminds'
# CLIENT_SECRET = 'your_client_secret'
REDIRECT_URI = 'https://www.hiveminds.in/'  



def flipkart_login_view(request):
    return render(request, 'auth.html', {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI
    })



@csrf_exempt
def exchange_token_view(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            oauth_code = body.get('code')

            if not oauth_code:
                return JsonResponse({'error': 'Missing OAuth code'}, status=400)


            token_url = 'https://ads.api.flipkart.net/ads-agency/token'
            payload = {
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'grant_type': 'authorization_code',
                'code': oauth_code,
                'redirect_uri': REDIRECT_URI,
                'scope': 'reporting campaign_management'
            }

            response = requests.post(token_url, data=payload)
            token_data = response.json()

            if response.status_code == 200:
                return JsonResponse(token_data)
            else:
                return JsonResponse({
                    'error': 'Failed to get token',
                    'details': token_data
                }, status=response.status_code)

        except Exception as e:
            return JsonResponse({'error': 'Internal error', 'message': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
#for lead scoring
import joblib

from django.http import JsonResponse
from rest_framework.decorators import api_view
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# Load model (only once)
# model = joblib.load(r'C:\Users\Administrator\Desktop\mindv3-20250706T051847Z-1-001\mindv3\OngoingProjects\MInD\MInD\adaboost_pipeline.pkl')
file_path = os.path.join(settings.BASE_DIR, 'MInD', 'adaboost_pipeline.pkl')

# Load the model
model = joblib.load(file_path)

@api_view(['POST'])
def predict_score(request):
    try:
        # Get input JSON
        input_data = request.data
        
        # Convert to DataFrame (single row)
        
        # import bigframes.pandas as bf



        # pd.set_option('display.max_columns', None)
        df = pd.DataFrame([input_data])
        df = df.drop(columns='income_proof')
        df_org = df.copy()
        print(df.columns)


        df.shape
        str_cols = ['user_pseudo_id', 'lead_id', 'nri_status', 'education', 'occupation', 'gender', 'smoker_status',
                    'income_range', 'city', 'utm_code', 'is_birthday_near', 'device_category', 'mobile_brand']

        num_cols = ['age', 'session_count_tilldate', 'diff_in_seconds', 'year_from_dob', 'month_from_dob', 'no_of_lead']

        for col in str_cols:
            df[col] = df[col].astype('str')
        for col in num_cols:
            df[col].fillna(np.nan, inplace = True)
            df[col] = df[col].astype('Int64')
        df['occupation'] = df['occupation'].replace(['na', '<NA>'], pd.NA)
        df['mobile_brand'] = df['mobile_brand'].replace(['na', '<NA>'], pd.NA)
        df['gender'] = df['gender'].replace(['na', '<NA>'], pd.NA)
        print(df.shape)
        # Drop actual NaN values
        df = df.dropna(subset=['occupation'])
        
        print(df.shape)
        # Remove non-English values using regex (keep only rows with a-z letters)
        # df = df[df['occupation'].str.contains('^[a-zA-Z]+$', regex=True)]
        
        print(df.shape)
        # df = df.dropna() #dropping null values since their percentage is less
        df = df[(df.age > 0) & (df.age < 100)]
        df = df[(df['diff_in_seconds'] > 0)]
        gender_map = {
            'm': 'm',
            'male': 'm',
            'f': 'f',
            'mx': 'm',
        }
        smoker_map = {
            'y': 'y',
            'yes': 'y',
            'n': 'n',
            'no': 'n',
        }

        df['gender'] = df['gender'].map(gender_map)
        df['city'] = df['city'].replace('Delhi','New Delhi')
        df['smoker_status'] = df['smoker_status'].map(smoker_map)
        df['lead_creation_date_and_time_ist'] = pd.to_datetime(df['lead_creation_date_and_time_ist'])

        df['lead_day'] = df['lead_creation_date_and_time_ist'].dt.day_name()
        df['lead_month'] = df['lead_creation_date_and_time_ist'].dt.month_name()
        df['lead_hour'] = df['lead_creation_date_and_time_ist'].dt.hour
        yes_no_encode = {
            'y': 1,
            'Y': 1,
            'n': 0,
            'N': 0
        }
        gender_encode = {
            'm': 1,
            'f': 0
        }

        df['nri_status'] = df['nri_status'].map(yes_no_encode)
        df['smoker_status'] = df['smoker_status'].map(yes_no_encode)
        df['is_birthday_near'] = df['is_birthday_near'].map(yes_no_encode)

        df['gender'] = df['gender'].map(gender_encode)
        df.rename(columns = {'gender': 'is_male'}, inplace = True)
        df['is_female']=1-df['is_male']
        edu_encode = {
            'illiterate': 0,
            'tenthpass': 1,
            'belowtwelvepass': 1,
            'highersecondary': 2,
            'twelvepass': 2,
            'graduate': 3,
            'graduateandabove': 3,
            'postgraduate': 3
        }
        income_encode = {
            '<3': 0,
            '3-5': 1,
            '5-7': 2,
            '7-10': 3,
            '>10': 4
        }

        df['education'] = df['education'].map(edu_encode)
        df['income_range'] = df['income_range'].map(income_encode)
        # Days of the week mapping
        day_encode = {
            'Monday': 1,
            'Tuesday': 2,
            'Wednesday': 3,
            'Thursday': 4,
            'Friday': 5,
            'Saturday': 6,
            'Sunday': 7
        }

        # Months of the year mapping
        month_encode = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }

        # Example: applying to dataframe columns
        df['lead_day'] = df['lead_day'].map(day_encode)
        df['lead_month'] = df['lead_month'].map(month_encode)
        occ_map = {
            'salaried': 'salaried',
            'professional': 'salaried',
            'selfemployed': 'selfemployed',
            'selfemployedfromhome': 'selfemployed',
            'housewife': 'unemployed',
            'homemaker': 'unemployed',
            'student': 'unemployed',
            'retired': 'unemployed',
            'others': 'others'
        }

        df['occupation'] = df['occupation'].map(occ_map)
        from sklearn.preprocessing import OneHotEncoder

        cols_to_encode = ['occupation', 'device_category']

        encoder = OneHotEncoder(drop = 'first')
        encoded_array = encoder.fit_transform(df[cols_to_encode])
        encoded_df = pd.DataFrame(encoded_array.toarray(), columns=encoder.get_feature_names_out(cols_to_encode))
        df_final = pd.concat([df.reset_index(drop = True).drop(columns=cols_to_encode), encoded_df], axis=1)

        
        

        top_10_brands = df_final['mobile_brand'].value_counts().nlargest(10).index
        print(top_10_brands)
        df_final['mobile_brand'] = df_final['mobile_brand'].where(df_final['mobile_brand'].isin(top_10_brands), 'Others')
        df_final.head()
        

        city_counts = df_final['city'].value_counts()
        total_cities = len(city_counts)

        tier1_cutoff = int(total_cities * 0.10)
        tier2_cutoff = int(total_cities * 0.30)
        tier3_cutoff = int(total_cities * 0.50)
        tier1_cities = city_counts.iloc[:tier1_cutoff].index
        tier2_cities = city_counts.iloc[tier1_cutoff:tier2_cutoff].index
        tier3_cities = city_counts.iloc[tier2_cutoff:tier3_cutoff].index
        def classify_city(city):
            if city in tier1_cities:
                return 'Tier 1'
            elif city in tier2_cities:
                return 'Tier 2'
            elif city in tier3_cities:
                return 'Tier 3'
            else:
                return 'Tier 4'

        # Apply the function
        df_final['City_Tier'] = df_final['city'].apply(classify_city)
        utm_mapping = {
            "14271449": "Top Funnel",
            "1111": "Bottom Funnel",
            "1311262": "Mid Funnel",
            "1311271": "Bottom Funnel",
            "14354532": "Netcore",
            "14298476": "Top Funnel",
            "1311279": "Bottom Funnel",
            "143712672": "Affiliate",
            "14271470": "Top Funnel",
            "143713611": "Top Funnel",
            "14364542": "Netcore",
            "143712623": "Affiliate",
            "143712660": "Affiliate",
            "14370713": "Affiliate",
            "143712678": "Affiliate",
            "143712664": "Affiliate",
            "143713615": "Affiliate",
            "143713626": "Affiliate",
            "14271458": "NRI",
            "1311265": "Mid Funnel",
            "14230408": "Top Funnel",
            "1337483705": "NRI",
            "143712626": "Affiliate",
            "143712650": "Affiliate",
            "143713681": "Top Funnel",
            "14363541": "Affiliate",
            "143713627": "Affiliate",
            "14353531": "Netcore",
            "14271450": "Top Funnel",
            "143713685": "Affiliate",
            "143712674": "Affiliate",
            "143712616": "Affiliate",
            "14271457": "Top Funnel",
            "1337493752": "Netcore",
            "1311273": "Bottom Funnel",
            "143712654": "Affiliate",
            "143712677": "Affiliate",
            "143712653": "Affiliate",
            "1311280": "Affiliate",
            "143712665": "Affiliate",
            "143712621": "Affiliate",
            "143712613": "Affiliate",
            "14271471": "Top Funnel",
            "143712679": "Affiliate",
            "143730681": "Affiliate",
            "143713625": "Affiliate",
            "143712666": "Affiliate",
            "1311277": "Bottom Funnel",
            "14298483": "Top Funnel",
            "143712668": "Affiliate",
            "143712614": "Affiliate",
            "143712667": "Affiliate",
            "14271455": "Affiliate",
            "143712655": "Affiliate",
            "143712658": "Affiliate",
            "143712676": "Affiliate",
            "14298477": "Top Funnel",
            "143712625": "Affiliate",
            "1419212": "Netcore",
            "143712622": "Affiliate",
            "143712656": "Affiliate",
            "143712689": "Affiliate",
            "1313289": "Mid Funnel",
            "143712628": "Affiliate",
            "143712612": "Affiliate",
            "143712611": "Affiliate",
            "143712663": "Affiliate",
            "143713621": "Affiliate",
            "143713684": "Top Funnel",
            "143713686": "Affiliate",
            "143730684": "Affiliate",
            "143712624": "Affiliate",
            "143712652": "Affiliate",
            "143712673": "Affiliate",
            "143713622": "Affiliate",
            "131909": "Mid Funnel",
            "14271481": "Top Funnel",
            "14370715": "Affiliate",
            "14371214": "Affiliate",
            "143712627": "Affiliate",
            "143712651": "Affiliate",
            "143712657": "Affiliate",
            "143712659": "Affiliate",
            "143712661": "Affiliate",
            "143712669": "Affiliate",
            "143712670": "Affiliate",
            "143712675": "Affiliate",
            "143712684": "Affiliate",
            "143712686": "Affiliate",
            '1337093709': 'Others',
            '14364543': 'Others',
            '1419213': 'Others',
            '1337493770': 'Others',
            '1234567': 'Others',
            '1116750': 'Others',
            '14354531': 'Others',
            '1337493771': 'Others',
            '1437137796': 'Others',
            '1116724': 'Others',
            '14230413': 'Others',
            '14271460': 'Others',
            '123456': 'Others',
            '1311269': 'Others',
            '1337493773': 'Others',
            '1337493753': 'Others',
            '1419215': 'Others',
            '6824LQAM': 'Others',
            '1412168': 'Others',
            '14364546': 'Others',
            '1437137795': 'Others',
            '14364544': 'Others',
            '1437137797': 'Others',
            '1': 'Others',
            '1116730': 'Others',
            '1116731': 'Others',
            '1337483708': 'Others',
            '14271482': 'Others',
            '14364545': 'Others',
            '14364547': 'Others'
        }
        df_final['utm_label'] = df_final['utm_code'].map(utm_mapping)
        df_final=df_final.drop(columns=['utm_code','user_pseudo_id','year_from_dob'])
        print(df_final)
        print("flag1")
        print(df_final.columns)
        expected_cols = model.feature_names_in_

# Add any missing columns with 0s
        for col in expected_cols:
            if col not in df_final.columns:
                df_final[col] = 0

        # Reorder columns to match training
        df_final = df_final[expected_cols]

        
        # Predict score (probability of being a lead)
        score = model.predict_proba(df_final)[:, 1][0]
        
        return JsonResponse({'score': round(float(score), 4)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


