from flask import Flask, render_template, request, jsonify, redirect, url_for, session, make_response
import secrets
from flask_sqlalchemy import SQLAlchemy
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import csv
from urllib.parse import urlparse
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import geopandas as gpd
import numpy as np
from geopy.distance import geodesic
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import traceback
import math
from collections import defaultdict
import openai
import schedule
#import time
#import threading
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure key

# Dummy admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

# Prevent caching after logout
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///businesses.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define Database Model
class Business(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_url = db.Column(db.String(500), nullable=False)


# Create Database Tables
with app.app_context():
    db.create_all()

# SEMrush API Key
SEMRUSH_API_KEY = "c08252868c1bd7ca6360254895cdecee"

# Open AI Key
openai.api_key = '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'

# Load Google Maps data (Assuming CSV has columns: 'latitude', 'longitude')
#df = pd.read_csv("c:\\LearnGit\\gmapsdemo\\comp_list_gmaps.csv")
csv_path = os.path.join(os.path.dirname(__file__), "comp_list_gmaps.csv")

# Read the CSV file
df = pd.read_csv(csv_path)

# Extract domain
def extract_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # Remove 'www.' if present
    return domain.replace("www.", "")

# Get Keywords
def get_keywords(url, database="us"):
    
        
        url = f"https://api.semrush.com/?type=domain_organic&key={SEMRUSH_API_KEY}&domain={url}&database=us&display_limit=10&export_columns=Ph,Po,PoPrev,PoDiff"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.text.splitlines()
            reader = csv.reader(data)
            headers = next(reader)  # Skip headers

            # Process data
            rank_changes = []
            for row in reader:
                if len(row) < 1:  # Ensure row has enough columns
                    print(f"Skipping row due to missing data: {row}")
                    continue  # Skip incomplete data
                
                # Convert to appropriate types
                try:
                    keyword = row[0].split(';')[0]
                    current_pos = row[0].split(';')[1]
                except ValueError as e:
                    print(f"Error parsing row {row}: {e}")
                    continue  # Skip invalid data

                rank_changes.append({
                    "keyword": keyword,
                    "current_position": current_pos
                })

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return rank_changes, timestamp

        else:
            return [], f"Error: {response.status_code}"
        
### Suggesting the best radius

def count_competitors(center, radius):
    count = 0
    for _, row in df.iterrows():
        dist = geodesic(center, (row["Latitude"], row["Longitude"])).miles
        if dist <= radius:
            count += 1
    return count

def find_hotspot_geocode(competitor_geocodes):
    """Cluster competitors and find the best geocode"""

    # Convert to NumPy array
    competitor_geocodes = np.array(competitor_geocodes)

    # Handle cases with too few competitors
    if len(competitor_geocodes) == 0:
        return (0, 0)  # Default value if no competitors are available
    elif len(competitor_geocodes) == 1:
        return competitor_geocodes[0]  # Return the only geocode available

    # Ensure the number of clusters is valid
    n_clusters = min(len(competitor_geocodes), 10)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(competitor_geocodes)

    # Get cluster centers (potential optimal locations)
    cluster_centers = kmeans.cluster_centers_

    # Find the cluster center closest to the mean geocode of all competitors
    hotspot_geocode = cluster_centers[np.argmin(
        np.linalg.norm(cluster_centers - competitor_geocodes.mean(axis=0), axis=1)
    )]

    return tuple(hotspot_geocode)  # Return as tuple (lat, lng)


def train_geocode_model(competitor_data):
    """Train an ML model to predict optimal geocode based on keyword ranking"""

    # Convert the list of dictionaries into a DataFrame
    competitor_data = pd.DataFrame(competitor_data)

    # Convert position values to numeric types in case they are being treated as strings
    competitor_data["current_position"] = pd.to_numeric(competitor_data["current_position"], errors="coerce")
    
    # Drop rows with missing values in position, lat, or lng
    competitor_data = competitor_data.dropna(subset=["lat", "lng", "current_position"])

    # Ensure there are valid data points to train on
    if len(competitor_data) < 2:
        print("Insufficient data for model training. Returning None.")
        return None

    # Compute the average position per (lat, lng) pair
    df = competitor_data.groupby(["lat", "lng"], as_index=False).agg(avg_position=("current_position", "mean"))

    # Prepare the training data
    X = df[["lat", "lng"]]
    y = df["avg_position"]

    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# Generate JSON-LD (Structured Data)
def generate_json_ld(business_name, business_address, lat, lng, business_type):
    # Ensure that latitudes and longitudes are regular Python floats (convert if needed)
    #optimized_lat = float(optimized_lat)
    #optimized_lng = float(optimized_lng)
    """Generate JSON-LD structured data with optimized geocode"""
    return {
        "@context": "https://schema.org",
        #"@type": "LocalBusiness",
        "@type": business_type,
        "name": business_name,
        "address": business_address,
        "geo": {
            "@type": "GeoCoordinates",
            "latitude": lat,
            "longitude": lng
        }
    }

def haversine(lat1, lng1, lat2, lng2):
    """Calculate the Haversine distance between two lat/lng points."""
    R = 6371  # Radius of the Earth in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c  # Distance in km
    return distance


def optimize_geocode(lat, lng, competitor_data, suggested_radius):
    """Optimize business geocode using ML"""

    business_lat, business_lng = float(lat), float(lng)  # Ensure float conversion

    # Extract competitor geocodes from the list
    competitor_geocodes = [(float(comp["lat"]), float(comp["lng"])) for comp in competitor_data]

    # Find optimal cluster center (hotspot)
    hotspot_lat, hotspot_lng = find_hotspot_geocode(competitor_geocodes)

    # Train ML model using competitor data
    model = train_geocode_model(competitor_data)

    # Predict geocode using the trained model
    predicted_geocode = model.predict([[business_lat, business_lng]])

    # Convert predicted values to float, ensuring index safety
    predicted_lat = float(predicted_geocode[0]) if len(predicted_geocode) > 0 else business_lat
    predicted_lng = float(predicted_geocode[1]) if len(predicted_geocode) > 1 else business_lng

    # Blend business location with hotspot & ML-predicted location
    optimized_lat = (business_lat + hotspot_lat + predicted_lat) / 3
    optimized_lng = (business_lng + hotspot_lng + predicted_lng) / 3

    # Convert np.float64 to regular float
    optimized_lat_float  = float(optimized_lat)
    optimized_lng_float  = float(optimized_lng)

    distance_from_business = haversine(business_lat, business_lng, optimized_lat_float, optimized_lng_float)

    # If the optimized location is outside the suggested radius, adjust it
    if distance_from_business > suggested_radius:
        # Calculate the scaling factor to bring it back within the radius
        scaling_factor = suggested_radius / distance_from_business
        adjusted_lat = business_lat + (optimized_lat_float - business_lat) * scaling_factor
        adjusted_lng = business_lng + (optimized_lng_float - business_lng) * scaling_factor

        # Update optimized geocode to adjusted location within the radius
        optimized_lat_float = adjusted_lat
        optimized_lng_float = adjusted_lng

    return {"lat": optimized_lat_float, "lng": optimized_lng_float}


##########

def save_competitor_data_to_csv(competitor_scrapping_data):
    """Save competitor scrapping data to a CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"competitor_data.csv"

    # Define the CSV headers
    headers = ["Name", "Average Rating", "Latitude", "Longitude", "Website", "Keyword", "Current Position", "Timestamp"]

    # Write data to CSV
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header row

        for competitor in competitor_scrapping_data:
            name = competitor["name"]
            avg_rating = competitor["avg_rating"]
            lat = competitor["lat"]
            lng = competitor["lng"]
            website = competitor["website"]
            timestamp = competitor["timestamp"]

            # Iterate through keywords and positions
            for keyword_data in competitor["keywords"]:
                writer.writerow([
                    name, avg_rating, lat, lng, website, 
                    keyword_data["keyword"], keyword_data["current_position"], timestamp
                ])

    print(f"Competitor data saved to {filename}")


##########

def read_competitor_data_from_csv(filename):
    """Read competitor data from a CSV file and store it in a list."""
    competitor_data = []

    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # Read CSV as dictionaries
        
        for row in reader:
            competitor_data.append({
                "name": row["Name"],
                "avg_rating": float(row["Average Rating"]),
                "lat": float(row["Latitude"]),
                "lng": float(row["Longitude"]),
                "website": row["Website"],
                "keyword": row["Keyword"],
                "current_position": int(row["Current Position"]),
                "timestamp": row["Timestamp"]
            })

    return competitor_data

def group_competitor_data(filename):
    competitor_data = defaultdict(lambda: {
        "website": None, 
        "avg_rating": None, 
        "keywords_positions": []
    })

    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # Read CSV as dictionaries

        for row in reader:
            name = row["Name"]
            keyword = row["Keyword"]
            position = int(row["Current Position"])
            website = row["Website"]
            avg_rating = float(row["Average Rating"])

            # Store website and rating (ensuring it's stored only once per competitor)
            competitor_data[name]["website"] = website
            competitor_data[name]["avg_rating"] = avg_rating

            # Append keyword and position as a tuple
            competitor_data[name]["keywords_positions"].append((keyword, position))

    return competitor_data  # Returns a dictionary with grouped data

### Generate related Keywords

def filter_best_keywords(keywords):
    # Remove generic or low-impact keywords
    ignore_list = {"marketing", "SEO", "digital", "services"}  # Customize as needed
    return [kw for kw in keywords if kw.lower() not in ignore_list]

def generate_related_keywords(business_keywords, new_keywords, business_name, business_address):
    # Join business_keywords and new_keywords into a single string to provide context for OpenAI
    #input_keywords = ', '.join(business_keywords) + '. New keywords based on these terms: ' + ', '.join(new_keywords)
    bus_keywords = ', '.join(business_keywords)
    new_kywrds = ', '.join(new_keywords)

    #prompt = f"{business_name} located on {business_address} has top SEO keywords are {bus_keywords}. Our competitor's top SEO keywords are {new_kywrds}. Provide a list of relevant and effective SEO keywords."

    prompt = f"""
    {business_name} located at {business_address} specializes in SEO. 
    Our top keywords: {bus_keywords}. 
    Our competitor's top keywords: {new_kywrds}. 
    Generate a list of the **most effective, high-volume, and conversion-optimized** SEO keywords. 
    Ensure they are **relevant, competitive, and localized** where possible.
    Provide the keywords **without numbering or extra text**—just the keywords separated by commas.
    """

    # Request to OpenAI to generate related keywords
    #response = openai.Completion.create(
    #    engine="text-davinci-003",
    #    prompt=prompt,
    #    max_tokens=100,
    #    n=1,
    #    stop=None,
    #    temperature=0.7,
    #)

    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a keyword research assistant focused on SEO optimization."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract suggested keywords
    related_keywords = openai_response['choices'][0]['message']['content'].strip().split(',')
    best_keywords = filter_best_keywords([kw.strip() for kw in related_keywords])

    # Clean and return the keywords as a list
    return best_keywords

######## DISPLAY MAP #############
# Create and Display Map
def create_map(lat, lng, competitors, business_name, business_address, o_lat, o_lng, suggested_radius):
    # Create a Folium map centered at the given latitude and longitude
    business_map = folium.Map(location=[lat, lng], zoom_start=12)

    suggested_radius_rounded_int = int(round(suggested_radius))

    radius_meters = suggested_radius_rounded_int * 1609.34  # Convert miles to meters

    # Add a circle for the radius
    folium.Circle(
        location=[o_lat, o_lng],
        radius=radius_meters,
        color='blue',
        fill=True,
        fill_opacity=0.2
    ).add_to(business_map)

    # Add main business marker
    popup_html = f"""
        <div style='max-width: 300px;'>
            <h4><b>{business_name}</b></h4>
            <p><b>Address: </b>{business_address}</p>
        </div>
    """

    folium.Marker(
        location=[lat, lng],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color='blue')
    ).add_to(business_map)

    # Create a MarkerCluster layer for competitor markers
    marker_cluster = MarkerCluster().add_to(business_map)

    # Add markers for each competitor
    for competitor in competitors:
        try:
            # Safely access dictionary keys with .get() to avoid KeyError
            name = competitor.get('Name', 'N/A')
            address = competitor.get('Fulladdress', 'N/A')
            latitude = float(competitor.get('Latitude', 0))
            longitude = float(competitor.get('Longitude', 0))

            # Create HTML content for the competitor popup
            popup_html_opt = f"""
                <div style='max-width: 300px;'>
                    <h4><b>{name}</b></h4>
                    <p><b>Address: </b>{address}</p>
                </div>
            """

            # Add a marker for the competitor
            folium.Marker(
                location=[latitude, longitude],
                popup=folium.Popup(popup_html_opt, max_width=300),
                icon=folium.Icon(color='red')
            ).add_to(marker_cluster)
        except KeyError as e:
            print(f"Missing key in competitor data: {e}")
        except ValueError as e:
            print(f"Invalid latitude/longitude for competitor {name}: {e}")

    # Add marker for the optimized geocode location (o_lat, o_lng)
    #popup_html_optimized = f"""
    #    <div style='max-width: 300px;'>
    #        <h4><b>Optimized Location</b></h4>
    #        <p><b>Latitude: </b>{o_lat}</p>
    #        <p><b>Longitude: </b>{o_lng}</p>
    #    </div>
    #"""

    #folium.Marker(
    #    location=[o_lat, o_lng],
    #    popup=folium.Popup(popup_html_optimized, max_width=300),
    #    icon=folium.Icon(color='green')  # Green for optimized location
    #).add_to(business_map)

    return business_map


# Routes
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect if session does not exist
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials, please try again!")
    
    return render_template('login.html', error=None)


#@app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/view-records', methods=['GET'])
def view_records():
    records = Business.query.all()
    return {"data": [{"id": r.id, "business_url": r.business_url} for r in records]}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:

        business_url = request.form['business_url']  # Get input from form
        new_entry = Business(business_url=business_url)

        db.session.add(new_entry)
        db.session.commit()
        #return "Business URL saved successfully!"

        ######## OWN BUSINESS DETAILS ######################
        #business_url = request.form.get('business_url')
        business_url = 'https://dental.metasensemarketing.com/'
        lat = '39.8823873'
        lng = '-75.0114454'
        business_name = "Dental Digital Marketing Agency"
        business_address = "1233 Berlin Rd, Unit 8, Suite 5, Voorhees, NJ 08043, USA"
        business_type = "Dental"

        domain = extract_domain(business_url)
        keywords_business, timestamp = get_keywords(domain)
        #keywords_business, timestamp = 1, 2

        ###########   BEST RADIUS #################

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Latitude, df.Longitude))
        #print(gdf)

        # Extract latitude and longitude
        X = np.array(list(zip(df["Latitude"], df["Longitude"])))

        # Apply K-Means Clustering
        num_clusters = 10  # Adjust based on data size
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)

        #print(df[["Latitude", "Longitude", "cluster"]])
        
        center_location = (lat, lng)

        for radius in range(0, 26):  # Covers all values from 0 to 25
            num_competitors = count_competitors(center_location, radius)
            #print(f"Radius: {radius} miles → Competitor Count: {num_competitors}")

        # Create dataset with geospatial features
        df["competitor_density"] = df["cluster"].map(df["cluster"].value_counts())
        df["search_volume"] = np.random.randint(500, 5000, size=len(df))  # Simulated search traffic data

        # Generate radius options dynamically from 0 to 25 miles
        radius_options = list(range(0, 26))

        # Define input features & target variable
        X = df[["competitor_density", "search_volume"]]
        y = np.random.choice(radius_options, size=len(df))  # Simulated "best radius" labels

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # Predict best radius for a new location
        new_location = pd.DataFrame({"competitor_density": [50], "search_volume": [3000]})
        predicted_radius = model.predict(new_location)

        suggested_radius = round(predicted_radius[0], 2)  # Round to 2 decimal places

        #print(f"Recommended radius: {predicted_radius[0]:.2f} miles")

        ######## COMPETITORS BUSINESS DETAILS ######################

        # Initialize an empty list to store the competitors
        competitors = []

        # Read the CSV file
        #with open('c:\\LearnGit\\gmapsdemo\\comp_list_gmaps.csv', mode='r', encoding='utf-8-sig') as file:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                competitors.append(row)

        # Competitor Scrapping
        competitor_scrapping_data = []  # List to store extracted data

        #save_competitor_data_to_csv(competitor_scrapping_data)

        filename = f"competitor_data.csv"  # Use the actual filename
        competitor_scrapping_data = read_competitor_data_from_csv(filename)

        # Optimize geocode using AI/ML
        optimized_geocode = optimize_geocode(lat, lng, competitor_scrapping_data, suggested_radius)

        ######## SUGGESTING KEYWORD ######################

        # Step 1: Add business-specific keywords
        business_keywords = []
        for entry in keywords_business:
            business_keywords.append(entry["keyword"].lower())

        # Step 2: Convert competitor keywords to a DataFrame
        comp_data = {
            "lat": [],
            "lng": [],
            "competitor_keywords": [],
            "position": []
        }

        for competitor in competitor_scrapping_data:
            #for keyword in competitor["keyword"]:
            comp_data["lat"].append(competitor['lat'])
            comp_data["lng"].append(competitor['lng'])
            comp_data["competitor_keywords"].append(competitor['keyword'].lower())  # Strip whitespace and convert to lowercase
            comp_data["position"].append(competitor['current_position'])

        df1 = pd.DataFrame(comp_data)

        # Step 3: Convert competitor keywords to numerical format using TF-IDF
        all_keywords = df1["competitor_keywords"].tolist() + business_keywords  # Merge business keywords
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(all_keywords)

        # Step 4: Apply K-Means clustering to group similar keywords
        # Step 4: Apply K-Means clustering to group similar keywords
        unique_keywords = df1["competitor_keywords"].unique()  # Find unique keywords to avoid duplicates
        n_clusters = min(10, len(unique_keywords))  # Set clusters based on unique keyword count

        if n_clusters == 1:
            print("Warning: Only 1 cluster found. Consider reviewing the dataset for more variability.")
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Use dynamic n_clusters
        df1["keyword_cluster"] = kmeans.fit_predict(X[:len(df1)])  # Only apply clustering to competitor keywords


        # Step 5: Define your target variable (position) and features (TF-IDF vectors and cluster)
        y = df1["position"]
        X_train, X_test, y_train, y_test = train_test_split(X[:len(df1)], y, test_size=0.2, random_state=42)

        # Step 6: Train an XGBoost model to predict keyword positions
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Step 7: Identify the best-performing clusters (lowest average position)
        df1['position'] = pd.to_numeric(df1['position'], errors='coerce')

        best_cluster = df1.groupby("keyword_cluster")["position"].mean().idxmin()

        # Step 8: Suggest new keywords by finding similar terms in the best-performing cluster
        best_cluster_keywords = df1[df1["keyword_cluster"] == best_cluster]["competitor_keywords"].values

        # Step 9: Using cosine similarity to find semantically similar keywords
        similar_keywords = []
        for keyword in best_cluster_keywords:
            # Compute cosine similarity between each keyword and the entire keyword list
            similarities = cosine_similarity(vectorizer.transform([keyword]), X)
            similar_indices = similarities.argsort()[0]
            similar_keywords.extend(all_keywords[i] for i in similar_indices)

        # Step 10: Remove duplicates and print out suggested new keywords
        suggested_keywords = list(set(similar_keywords))
        suggested_keywords = [kw for kw in suggested_keywords if kw not in business_keywords]  # Remove business keywords from suggestions

        # Step 11: Predict the best position for a new set of keywords
        new_keywords = suggested_keywords + business_keywords
        new_X = vectorizer.transform(new_keywords)
        predicted_positions = model.predict(new_X)

        # Prepare the data to send to the HTML template
        suggested_data = zip(new_keywords, predicted_positions)

        # Generate related keywords
        related_keywords = generate_related_keywords(business_keywords, new_keywords, business_name, business_address)


        ######## GENERATE STRUCTURED DATA ######################
        #structured_data = generate_json_ld(business_name, business_address, optimized_geocode["lat"], optimized_geocode["lng"])
        structured_data = generate_json_ld(business_name, business_address, lat, lng, business_type)

        grouped_data = group_competitor_data(filename)

        # Create map
        business_map = create_map(lat, lng, competitors, business_name, business_address, optimized_geocode["lat"], optimized_geocode["lng"], suggested_radius)
        map_file = 'static/files/map.html'
        business_map.save(map_file)

        return render_template('results.html', business_name=business_name, keywords_business=keywords_business,
                               competitor_scrapping_data=competitor_scrapping_data, map_file="map.html",
                               suggested_radius=suggested_radius, structured_data=structured_data, 
                               grouped_data=grouped_data, suggested_data=suggested_data, related_keywords=related_keywords)
                               
    except Exception as e:
        print(f"Error from Main: {e.with_traceback()}")
        traceback.print_exc()
        return jsonify({"error": "An error occurred during analysis."})

@app.route('/logout')
def logout():
    session.clear()  # Clear session completely
    return redirect(url_for('login'))
    #session.pop('username', None)
    

    
# 3️⃣ Schedule the job to run every day at midnight
#schedule.every().day.at("01:05").do(analyze)

# 4️⃣ Function to continuously check & run scheduled tasks
#def run_scheduler():
#    print("🟢 Scheduler Started... Checking for tasks every 60 seconds.")
#    while True:
#        schedule.run_pending()
#        time.sleep(60)  # Wait 1 minute before checking again

# 5️⃣ Start the scheduler in a background thread when the app starts
#scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
#scheduler_thread.start()

# Run the Flask app
if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run()