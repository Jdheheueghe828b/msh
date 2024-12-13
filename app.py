from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import yaml
import os
import io
import base64
import cv2
import random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the food database
FOOD_DATABASE = {
    # الخبز والمخبوزات - Breads & Pastries
    'bread': {'name_ar': 'خبز أبيض', 'carbs': 15, 'serving_size': 'شريحة واحدة'},
    'arabic_bread': {'name_ar': 'خبز عربي', 'carbs': 20, 'serving_size': 'رغيف واحد'},
    'samoon': {'name_ar': 'صمون عراقي', 'carbs': 28, 'serving_size': 'قطعة واحدة'},
    'manakish': {'name_ar': 'مناقيش', 'carbs': 33, 'serving_size': 'قطعة واحدة'},
    'croissant': {'name_ar': 'كرواسون', 'carbs': 26, 'serving_size': 'قطعة واحدة'},
    'kaak': {'name_ar': 'كعك', 'carbs': 23, 'serving_size': 'قطعة واحدة'},
    'shrak_bread': {'name_ar': 'خبز شراك', 'carbs': 18, 'serving_size': 'رغيف واحد'},
    'tandoor_bread': {'name_ar': 'خبز تنور', 'carbs': 22, 'serving_size': 'رغيف واحد'},

    # الأرز والحبوب - Rice & Grains
    'white_rice': {'name_ar': 'أرز أبيض', 'carbs': 45, 'serving_size': 'كوب مطبوخ'},
    'brown_rice': {'name_ar': 'أرز بني', 'carbs': 42, 'serving_size': 'كوب مطبوخ'},
    'yellow_rice': {'name_ar': 'أرز بالزعفران', 'carbs': 45, 'serving_size': 'كوب مطبوخ'},
    'mujaddara': {'name_ar': 'مجدرة', 'carbs': 40, 'serving_size': 'كوب مطبوخ'},
    'freekeh': {'name_ar': 'فريكة', 'carbs': 35, 'serving_size': 'كوب مطبوخ'},
    'bulgur': {'name_ar': 'برغل', 'carbs': 34, 'serving_size': 'كوب مطبوخ'},
    'couscous': {'name_ar': 'كسكس', 'carbs': 36, 'serving_size': 'كوب مطبوخ'},
    'vermicelli_rice': {'name_ar': 'أرز بالشعيرية', 'carbs': 47, 'serving_size': 'كوب مطبوخ'},

    # اللحوم والدجاج - Meat & Chicken Dishes
    'shawarma_chicken': {'name_ar': 'شاورما دجاج', 'carbs': 30, 'serving_size': 'ساندويتش متوسط'},
    'shawarma_meat': {'name_ar': 'شاورما لحم', 'carbs': 28, 'serving_size': 'ساندويتش متوسط'},
    'grilled_chicken': {'name_ar': 'دجاج مشوي', 'carbs': 0, 'serving_size': 'صدر دجاج'},
    'shish_tawook': {'name_ar': 'شيش طاووق', 'carbs': 5, 'serving_size': 'سيخ واحد'},
    'kofta': {'name_ar': 'كفتة', 'carbs': 8, 'serving_size': 'سيخ واحد'},
    'kebab': {'name_ar': 'كباب', 'carbs': 6, 'serving_size': 'سيخ واحد'},
    'chicken_tikka': {'name_ar': 'دجاج تكا', 'carbs': 4, 'serving_size': 'قطعتين'},
    'grilled_liver': {'name_ar': 'كبدة مشوية', 'carbs': 3, 'serving_size': 'حصة متوسطة'},

    # الأطباق الرئيسية - Main Dishes
    'mansaf': {'name_ar': 'منسف', 'carbs': 65, 'serving_size': 'طبق متوسط'},
    'mandi': {'name_ar': 'مندي', 'carbs': 55, 'serving_size': 'طبق متوسط'},
    'maqluba': {'name_ar': 'مقلوبة', 'carbs': 48, 'serving_size': 'طبق متوسط'},
    'biryani': {'name_ar': 'برياني', 'carbs': 58, 'serving_size': 'طبق متوسط'},
    'kabsa': {'name_ar': 'كبسة', 'carbs': 52, 'serving_size': 'طبق متوسط'},
    'molokhia': {'name_ar': 'ملوخية', 'carbs': 15, 'serving_size': 'طبق متوسط'},
    'bamia': {'name_ar': 'بامية', 'carbs': 18, 'serving_size': 'طبق متوسط'},
    'fasolia': {'name_ar': 'فاصوليا خضراء', 'carbs': 20, 'serving_size': 'طبق متوسط'},
    'stuffed_grape_leaves': {'name_ar': 'ورق عنب', 'carbs': 25, 'serving_size': '5 قطع'},
    'stuffed_zucchini': {'name_ar': 'كوسا محشي', 'carbs': 22, 'serving_size': '2 حبات'},

    # المقبلات والسلطات - Appetizers & Salads
    'hummus': {'name_ar': 'حمص', 'carbs': 35, 'serving_size': 'كوب واحد'},
    'mutabal': {'name_ar': 'متبل', 'carbs': 20, 'serving_size': 'كوب واحد'},
    'tabbouleh': {'name_ar': 'تبولة', 'carbs': 25, 'serving_size': 'كوب واحد'},
    'fattoush': {'name_ar': 'فتوش', 'carbs': 15, 'serving_size': 'طبق متوسط'},
    'baba_ganoush': {'name_ar': 'بابا غنوج', 'carbs': 18, 'serving_size': 'كوب واحد'},
    'muhammara': {'name_ar': 'محمرة', 'carbs': 22, 'serving_size': 'كوب واحد'},
    'shanklish': {'name_ar': 'شنكليش', 'carbs': 8, 'serving_size': 'كوب واحد'},
    'labneh': {'name_ar': 'لبنة', 'carbs': 7, 'serving_size': 'كوب واحد'},

    # الحلويات الشرقية - Oriental Sweets
    'baklava': {'name_ar': 'بقلاوة', 'carbs': 40, 'serving_size': 'قطعة واحدة'},
    'kunafa': {'name_ar': 'كنافة', 'carbs': 50, 'serving_size': 'قطعة متوسطة'},
    'znoud_elsit': {'name_ar': 'زنود الست', 'carbs': 45, 'serving_size': 'قطعة واحدة'},
    'osmalia': {'name_ar': 'عثملية', 'carbs': 42, 'serving_size': 'قطعة واحدة'},
    'qatayef': {'name_ar': 'قطايف', 'carbs': 38, 'serving_size': 'قطعتين'},
    'luqaimat': {'name_ar': 'لقيمات', 'carbs': 35, 'serving_size': '5 حبات'},
    'halawet_eljibn': {'name_ar': 'حلاوة الجبن', 'carbs': 44, 'serving_size': 'قطعتين'},
    'maamoul': {'name_ar': 'معمول', 'carbs': 30, 'serving_size': 'قطعة واحدة'},
    'basbousa': {'name_ar': 'بسبوسة', 'carbs': 48, 'serving_size': 'قطعة متوسطة'},
    'umm_ali': {'name_ar': 'أم علي', 'carbs': 52, 'serving_size': 'طبق صغير'},

    # المشروبات - Beverages
    'arabic_coffee': {'name_ar': 'قهوة عربية', 'carbs': 0, 'serving_size': 'فنجان'},
    'turkish_coffee': {'name_ar': 'قهوة تركية', 'carbs': 1, 'serving_size': 'فنجان'},
    'karak_tea': {'name_ar': 'شاي كرك', 'carbs': 15, 'serving_size': 'كوب متوسط'},
    'mint_tea': {'name_ar': 'شاي بالنعناع', 'carbs': 0, 'serving_size': 'كوب متوسط'},
    'sahlab': {'name_ar': 'سحلب', 'carbs': 35, 'serving_size': 'كوب متوسط'},
    'qamar_al_din': {'name_ar': 'قمر الدين', 'carbs': 32, 'serving_size': 'كوب متوسط'},
    'jallab': {'name_ar': 'جلاب', 'carbs': 30, 'serving_size': 'كوب متوسط'},
    'fresh_juice_mix': {'name_ar': 'عصير طازج مشكل', 'carbs': 28, 'serving_size': 'كوب متوسط'},

    # الفطور - Breakfast Items
    'foul_medames': {'name_ar': 'فول مدمس', 'carbs': 35, 'serving_size': 'طبق متوسط'},
    'shakshuka': {'name_ar': 'شكشوكة', 'carbs': 12, 'serving_size': 'طبق متوسط'},
    'fatteh': {'name_ar': 'فتة', 'carbs': 45, 'serving_size': 'طبق متوسط'},
    'zaatar_manakish': {'name_ar': 'مناقيش زعتر', 'carbs': 32, 'serving_size': 'قطعة واحدة'},
    'cheese_manakish': {'name_ar': 'مناقيش جبنة', 'carbs': 30, 'serving_size': 'قطعة واحدة'},
    'eggs_with_sumac': {'name_ar': 'بيض بالسماق', 'carbs': 2, 'serving_size': 'طبق متوسط'},
    'makdous': {'name_ar': 'مكدوس', 'carbs': 8, 'serving_size': '3 حبات'},
    'zeit_and_zaatar': {'name_ar': 'زيت وزعتر', 'carbs': 5, 'serving_size': 'ملعقة كبيرة'},

    # المعجنات - Pastries
    'spinach_fatayer': {'name_ar': 'فطاير سبانخ', 'carbs': 28, 'serving_size': 'قطعتين'},
    'cheese_fatayer': {'name_ar': 'فطاير جبنة', 'carbs': 26, 'serving_size': 'قطعتين'},
    'meat_fatayer': {'name_ar': 'فطاير لحمة', 'carbs': 25, 'serving_size': 'قطعتين'},
    'zaatar_bread': {'name_ar': 'خبز زعتر', 'carbs': 30, 'serving_size': 'قطعة متوسطة'},
    'sfiha': {'name_ar': 'صفيحة', 'carbs': 22, 'serving_size': '2 قطع'},
    'cheese_rolls': {'name_ar': 'رقاقات جبنة', 'carbs': 24, 'serving_size': '2 قطع'},
    'sambousek': {'name_ar': 'سمبوسك', 'carbs': 20, 'serving_size': '2 قطع'},
    'pizza_arabic': {'name_ar': 'بيتزا عربي', 'carbs': 35, 'serving_size': 'قطعتين'},

    # الأسماك والمأكولات البحرية - Fish & Seafood
    'grilled_fish': {'name_ar': 'سمك مشوي', 'carbs': 0, 'serving_size': 'حصة متوسطة'},
    'sayyadieh': {'name_ar': 'صيادية', 'carbs': 42, 'serving_size': 'طبق متوسط'},
    'fish_sayadiya': {'name_ar': 'سمك صيادية', 'carbs': 40, 'serving_size': 'طبق متوسط'},
    'shrimp_rice': {'name_ar': 'أرز بالجمبري', 'carbs': 45, 'serving_size': 'طبق متوسط'},
    'calamari': {'name_ar': 'كاليماري مقلي', 'carbs': 15, 'serving_size': 'حصة متوسطة'},
    'grilled_shrimp': {'name_ar': 'جمبري مشوي', 'carbs': 0, 'serving_size': '6 حبات'},
    'fish_tagine': {'name_ar': 'طاجن سمك', 'carbs': 25, 'serving_size': 'طبق متوسط'},
    'samak_harra': {'name_ar': 'سمك حرة', 'carbs': 8, 'serving_size': 'حصة متوسطة'}
}

# Load TensorFlow model
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2"
model = hub.load(model_url)

class FoodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(100), nullable=False)
    carbs = db.Column(db.Float, nullable=False)
    serving_size = db.Column(db.String(50))
    portion_multiplier = db.Column(db.Float, default=1.0)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    confidence = db.Column(db.Float)
    meal_type = db.Column(db.String(20))  # فطور، غداء، عشاء
    image_hash = db.Column(db.String(64))

class NutritionTips(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(100))
    tip_ar = db.Column(db.String(500))
    category = db.Column(db.String(50))

PORTION_SIZES = {
    'صغير': 0.5,
    'عادي': 1.0,
    'كبير': 1.5,
    'كبير جداً': 2.0
}

MEAL_TYPES = {
    'morning': 'فطور',
    'afternoon': 'غداء',
    'evening': 'عشاء',
    'snack': 'وجبة خفيفة'
}

NUTRITION_TIPS = {
    'high_carbs': [
        'تناول هذا الطعام مع البروتين لإبطاء امتصاص السكر',
        'يفضل تناول نصف الكمية مع الخضروات',
        'مناسب بعد التمرين الرياضي'
    ],
    'moderate_carbs': [
        'كمية معتدلة من الكربوهيدرات، مناسبة للوجبات الرئيسية',
        'يمكن تناوله مع السلطة لوجبة متوازنة',
        'مثالي للطاقة المستدامة'
    ],
    'low_carbs': [
        'خيار جيد للحمية منخفضة الكربوهيدرات',
        'يمكن تناوله بحرية نسبياً',
        'مناسب للوجبات الخفيفة'
    ]
}

def get_nutrition_tip(carbs_value):
    if carbs_value > 40:
        return random.choice(NUTRITION_TIPS['high_carbs'])
    elif carbs_value > 20:
        return random.choice(NUTRITION_TIPS['moderate_carbs'])
    else:
        return random.choice(NUTRITION_TIPS['low_carbs'])

def get_meal_type():
    hour = datetime.now().hour
    if 5 <= hour < 11:
        return 'فطور'
    elif 11 <= hour < 16:
        return 'غداء'
    elif 16 <= hour < 22:
        return 'عشاء'
    else:
        return 'وجبة خفيفة'

def enhance_image(image):
    """تحسين جودة الصورة للتعرف الأفضل"""
    # Convert to array
    img_array = np.array(image)
    
    # Enhance contrast
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Denoise
    img_output = cv2.fastNlMeansDenoisingColored(img_output)
    
    return Image.fromarray(img_output)

def detect_multiple_foods(image):
    """التعرف على عدة أطعمة في الصورة الواحدة"""
    results = []
    img_array = np.array(image)
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find edges
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore very small regions
            x, y, w, h = cv2.boundingRect(contour)
            food_region = image.crop((x, y, x+w, y+h))
            results.append(food_region)
    
    return results

def estimate_portion_size(image):
    """تقدير حجم الحصة باستخدام تحليل الصورة"""
    width, height = image.size
    total_pixels = width * height
    
    # Convert to array for analysis
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        food_area = cv2.contourArea(largest_contour)
        
        # Calculate ratio of food area to total image area
        area_ratio = food_area / total_pixels
        
        if area_ratio > 0.7:
            return 'كبير جداً'
        elif area_ratio > 0.5:
            return 'كبير'
        elif area_ratio > 0.3:
            return 'عادي'
        else:
            return 'صغير'
    
    return 'عادي'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_data = request.json['image']
        img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
        
        # Enhance image quality
        enhanced_img = enhance_image(img)
        
        # Detect multiple foods
        food_regions = detect_multiple_foods(enhanced_img)
        
        all_results = []
        for region in food_regions:
            # Prepare image for model
            region = region.resize((224, 224))
            img_array = np.array(region) / 255.0
            img_array = np.expand_dims(img_array, 0)
            
            # Get predictions
            predictions = model(img_array)
            predictions = tf.nn.softmax(predictions).numpy()
            
            # Get top prediction
            predicted_class = tf.argmax(predictions[0]).numpy()
            confidence = float(predictions[0][predicted_class])
            
            # Estimate portion size
            portion_size = estimate_portion_size(region)
            portion_multiplier = PORTION_SIZES[portion_size]
            
            # Get meal type based on time
            meal_type = get_meal_type()
            
            # Match with database
            for food_name, food_info in FOOD_DATABASE.items():
                if food_name in str(predicted_class).lower():
                    # Calculate adjusted carbs based on portion
                    adjusted_carbs = food_info['carbs'] * portion_multiplier
                    
                    # Get nutrition tip
                    nutrition_tip = get_nutrition_tip(adjusted_carbs)
                    
                    result = {
                        'food_name': food_info['name_ar'],
                        'carbs': adjusted_carbs,
                        'base_carbs': food_info['carbs'],
                        'serving_size': food_info['serving_size'],
                        'portion_size': portion_size,
                        'confidence': round(confidence * 100, 1),
                        'meal_type': meal_type,
                        'nutrition_tip': nutrition_tip
                    }
                    
                    # Save to database
                    food_log = FoodLog(
                        food_name=food_info['name_ar'],
                        carbs=adjusted_carbs,
                        serving_size=food_info['serving_size'],
                        portion_multiplier=portion_multiplier,
                        confidence=confidence,
                        meal_type=meal_type
                    )
                    db.session.add(food_log)
                    all_results.append(result)
                    break
        
        db.session.commit()
        return jsonify({
            'success': True,
            'results': all_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def get_stats():
    try:
        # Get today's date
        today = datetime.now().date()
        
        # Get all logs for today
        today_logs = FoodLog.query.filter(
            db.func.date(FoodLog.timestamp) == today
        ).all()
        
        # Calculate stats by meal type
        meal_stats = {
            'فطور': {'carbs': 0, 'items': []},
            'غداء': {'carbs': 0, 'items': []},
            'عشاء': {'carbs': 0, 'items': []},
            'وجبة خفيفة': {'carbs': 0, 'items': []}
        }
        
        total_carbs = 0
        for log in today_logs:
            meal_stats[log.meal_type]['carbs'] += log.carbs
            meal_stats[log.meal_type]['items'].append({
                'food_name': log.food_name,
                'carbs': log.carbs,
                'serving_size': log.serving_size,
                'time': log.timestamp.strftime('%H:%M')
            })
            total_carbs += log.carbs
        
        # Get weekly stats
        week_ago = today - timedelta(days=7)
        weekly_logs = FoodLog.query.filter(
            db.func.date(FoodLog.timestamp) >= week_ago
        ).all()
        
        weekly_stats = {}
        for log in weekly_logs:
            date = log.timestamp.date().strftime('%Y-%m-%d')
            if date not in weekly_stats:
                weekly_stats[date] = 0
            weekly_stats[date] += log.carbs
        
        return jsonify({
            'success': True,
            'today': {
                'total_carbs': total_carbs,
                'meal_stats': meal_stats
            },
            'weekly_stats': weekly_stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def get_history():
    logs = FoodLog.query.order_by(FoodLog.timestamp.desc()).limit(10).all()
    history = []
    for log in logs:
        history.append({
            'food_name': log.food_name,
            'carbs': log.carbs,
            'serving_size': log.serving_size,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify({'history': history})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
