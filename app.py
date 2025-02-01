from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS

# Robust import handling for SQLAlchemy
try:
    from flask_sqlalchemy import SQLAlchemy
except ImportError:
    print("Flask-SQLAlchemy not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask-sqlalchemy'])
    from flask_sqlalchemy import SQLAlchemy

try:
    from flask_migrate import Migrate
except ImportError:
    print("Flask-Migrate not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask-migrate'])
    from flask_migrate import Migrate

# Robust email validation
try:
    import email_validator
except ImportError:
    print("email-validator not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'email-validator'])
    import email_validator

import ollama
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import requests
import json
import traceback
import re
from datetime import datetime, timedelta

# Import our models
from models import db, User, Workout, WorkoutStatistic, init_db, BiometricData, RecoveryLog, WorkoutLog

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///fitness_tracker.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')

# Initialize database
migrate = Migrate(app, db)
init_db(app)

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'workout_generator_full.log'
file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=1024 * 1024 * 10,  # 10 MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,  # File handler for persistent logging
        logging.StreamHandler()  # Console handler
    ]
)

# Email validation function
def validate_email(email):
    """
    Validate email address with fallback mechanism
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email is valid, False otherwise
    """
    try:
        # Try email_validator first
        email_validator.validate_email(email)
        return True
    except (email_validator.EmailNotValidError, ImportError):
        # Fallback to simple regex validation
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, email) is not None

def generate_workout_with_ollama(prompt, base_url, api_key=None, model='llama3.2:latest'):
    """
    Generate workout plan using Ollama API with enhanced error handling
    """
    try:
        import requests
        
        # Validate inputs
        if not base_url:
            raise ValueError("Ollama base URL is required")
        
        # Construct full URL
        url = f"{base_url.rstrip('/')}/api/generate"
        
        # Log detailed connection info
        logging.info(f"Attempting to generate workout with:")
        logging.info(f"  Base URL: {url}")
        logging.info(f"  Model: {model}")
        logging.info(f"  Prompt Length: {len(prompt)} characters")
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Temporarily disable streaming for debugging
            "options": {
                "temperature": 0.7
            }
        }
        
        # Add API key if provided
        headers = {
            'Content-Type': 'application/json'
        }
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Make request with extended timeout and detailed logging
        try:
            logging.info("Sending request to Ollama...")
            response = requests.post(url, 
                                     json=payload, 
                                     headers=headers, 
                                     timeout=120)  # 2 minutes timeout
            
            # Log full response details
            logging.info(f"Response Status Code: {response.status_code}")
            logging.debug(f"Response Headers: {response.headers}")
            
            # Check response
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Log response details
            logging.info("Response received successfully")
            logging.debug(f"Full response: {result}")
            
            # Extract generated text
            generated_text = result.get('response', '')
            
            if not generated_text:
                logging.warning("Empty response received from Ollama")
                raise ValueError("No text generated by the model")
            
            logging.info(f"Generated text length: {len(generated_text)} characters")
            
            return generated_text
        
        except requests.exceptions.RequestException as req_error:
            logging.error(f"Request Error: {req_error}")
            logging.error(f"Error Details: {traceback.format_exc()}")
            raise ValueError(f"Network error connecting to Ollama: {req_error}")
    
    except Exception as e:
        logging.error(f"Unexpected error in workout generation: {e}")
        logging.error(traceback.format_exc())
        raise

def parse_workout_plan(raw_text):
    """
    Parse the raw text into a structured workout plan with enhanced flexibility
    """
    try:
        import re
        import json
        
        # Multiple parsing strategies
        parsing_strategies = [
            # Strategy 1: Direct JSON extraction
            lambda text: json.loads(text),
            
            # Strategy 2: Extract JSON between specific markers
            lambda text: json.loads(re.search(r'\{.*"metadata".*"daily_plan".*\}', text, re.DOTALL | re.IGNORECASE).group(0)),
            
            # Strategy 3: Clean and parse JSON
            lambda text: json.loads(re.sub(r'^[^{]*', '', re.sub(r'[^}]*$', '', text)))
        ]
        
        # Try each parsing strategy
        for strategy in parsing_strategies:
            try:
                parsed_plan = strategy(raw_text)
                
                # Validate parsed plan structure
                if not isinstance(parsed_plan, dict):
                    raise ValueError("Invalid workout plan structure")
                
                # Ensure required keys exist
                required_keys = ['metadata', 'daily_plan']
                for key in required_keys:
                    if key not in parsed_plan:
                        raise KeyError(f"Missing required key: {key}")
                
                # Validate daily plan structure
                daily_plan = parsed_plan['daily_plan'][0]
                required_sections = ['warmup', 'main_workout', 'cooldown']
                for section in required_sections:
                    if section not in daily_plan:
                        raise KeyError(f"Missing required section: {section}")
                
                # Enhance exercises with default values if missing
                for section in required_sections:
                    for exercise in daily_plan.get(section, []):
                        # Add default values
                        exercise.setdefault('sets', 3)
                        exercise.setdefault('reps', 10)
                        exercise.setdefault('description', 'No description available')
                        
                        # Validate exercise structure
                        if 'name' not in exercise:
                            exercise['name'] = 'Unnamed Exercise'
                
                return parsed_plan
            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.warning(f"Parsing strategy failed: {e}")
                continue
        
        # If all strategies fail
        raise ValueError("Unable to parse workout plan from generated text")
    
    except Exception as e:
        logging.error(f"Comprehensive parsing error: {e}")
        logging.error(f"Raw text received: {raw_text}")
        raise

def get_user_workout_history(user_id, weeks_to_retrieve=4):
    """
    Retrieve user's workout history for context-aware workout generation
    
    Args:
        user_id (int): ID of the user
        weeks_to_retrieve (int): Number of weeks of workout history to retrieve
    
    Returns:
        dict: Summarized workout history with key insights
    """
    from datetime import datetime, timedelta
    
    # Calculate the date threshold
    date_threshold = datetime.utcnow() - timedelta(weeks=weeks_to_retrieve)
    
    # Query recent workouts
    recent_workouts = Workout.query.filter(
        Workout.user_id == user_id,
        Workout.date >= date_threshold
    ).order_by(Workout.date.desc()).all()
    
    # Analyze workout statistics
    workout_stats = WorkoutStatistic.query.filter(
        WorkoutStatistic.user_id == user_id,
        WorkoutStatistic.completed_date >= date_threshold
    ).all()
    
    # Summarize workout insights
    workout_summary = {
        'total_workouts': len(recent_workouts),
        'average_perceived_difficulty': None,
        'exercises_completed': 0,
        'total_exercises': 0,
        'workout_types': set(),
        'fitness_goals': set(),
        'health_conditions': None,
        'progression_notes': []
    }
    
    # Aggregate workout insights
    if workout_stats:
        difficulties = [stat.perceived_difficulty for stat in workout_stats if stat.perceived_difficulty]
        workout_summary['average_perceived_difficulty'] = (
            sum(difficulties) / len(difficulties) if difficulties else None
        )
        
        workout_summary['exercises_completed'] = sum(stat.exercises_completed for stat in workout_stats)
        workout_summary['total_exercises'] = sum(stat.total_exercises for stat in workout_stats)
    
    # Extract workout types and fitness goals
    for workout in recent_workouts:
        workout_plan = workout.get_workout_plan()
        if workout_plan:
            workout_summary['workout_types'].add(
                workout_plan.get('metadata', {}).get('difficulty', 'Unspecified')
            )
            workout_summary['fitness_goals'].update(
                workout_plan.get('metadata', {}).get('fitness_goals', [])
            )
    
    # Get user profile for additional context
    user = User.query.get(user_id)
    if user:
        workout_summary['health_conditions'] = user.health_conditions
    
    # Basic progression analysis
    if len(recent_workouts) > 1:
        progression_analysis = analyze_workout_progression(recent_workouts)
        workout_summary['progression_notes'] = progression_analysis
    
    return workout_summary

def analyze_workout_progression(workouts):
    """
    Analyze progression and provide insights for workout generation
    
    Args:
        workouts (list): List of recent workouts
    
    Returns:
        list: Progression insights and recommendations
    """
    progression_notes = []
    
    # Check workout difficulty progression
    difficulties = [
        workout.get_workout_plan().get('metadata', {}).get('difficulty', 'Unspecified') 
        for workout in workouts
    ]
    
    # Basic difficulty tracking
    if len(set(difficulties)) == 1:
        progression_notes.append(
            "Consistent difficulty level detected. Consider gradual intensity increase."
        )
    elif difficulties[0] != difficulties[-1]:
        progression_notes.append(
            f"Difficulty progression from {difficulties[0]} to {difficulties[-1]} observed."
        )
    
    # Exercise type diversity
    exercise_types = [
        set(workout.get_workout_plan().get('daily_plan', [{}])[0].get('main_workout', []))
        for workout in workouts
    ]
    
    if len(exercise_types) > 1 and len(set(map(frozenset, exercise_types))) == 1:
        progression_notes.append(
            "Repetitive exercise types detected. Recommend introducing exercise variety."
        )
    
    return progression_notes

def generate_workout_plan(user_id=None):
    """
    Generate a personalized workout plan with database-driven context
    
    Args:
        user_id (int, optional): User ID to generate personalized workout
    
    Returns:
        dict: Generated workout plan
    """
    # If no user_id is provided, try to get from session
    if user_id is None and 'user_id' in session:
        user_id = session['user_id']
    
    # Workout generation context
    context = {
        "base_prompt": "Generate a personalized fitness workout plan",
        "user_insights": None
    }
    
    # Retrieve user workout history if user_id is available
    if user_id:
        try:
            user_insights = get_user_workout_history(user_id)
            context['user_insights'] = user_insights
            
            # Enhance base prompt with user insights
            context['base_prompt'] += f" for a user with the following profile: {json.dumps(user_insights)}"
        except Exception as e:
            logging.error(f"Error retrieving user workout history: {e}")
    
    # Use existing workout generation logic with enhanced context
    try:
        workout_plan = generate_workout_with_ollama(
            context['base_prompt'], 
            os.getenv('OLLAMA_BASE_URL'), 
            model=os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
        )
        
        # If user_id is available, save the generated workout
        if user_id:
            save_generated_workout(user_id, workout_plan)
        
        return workout_plan
    
    except Exception as e:
        logging.error(f"Workout generation error: {e}")
        raise

def save_generated_workout(user_id, workout_plan):
    """
    Save generated workout to the database
    
    Args:
        user_id (int): User ID
        workout_plan (dict): Generated workout plan
    """
    new_workout = Workout(
        user_id=user_id,
        difficulty=workout_plan.get('metadata', {}).get('difficulty', 'Not specified'),
        total_duration=workout_plan.get('metadata', {}).get('total_duration', 'Not specified'),
        fitness_goals=', '.join(workout_plan.get('metadata', {}).get('fitness_goals', []))
    )
    new_workout.set_workout_plan(workout_plan)
    
    db.session.add(new_workout)
    db.session.commit()
    
    logging.info(f"Workout saved for user {user_id}")

def generate_workout_plan():
    """
    Generate a personalized workout plan using Ollama
    """
    # Log the start of the request processing
    logging.error("Starting workout plan generation")
    logging.error(f"Request method: {request.method}")
    
    try:
        # Comprehensive request logging
        logging.error(f"Request content type: {request.content_type}")
        logging.error(f"Request data type: {type(request.data)}")
        logging.error(f"Request data (raw): {request.data}")
        
        # Get input parameters with extensive error handling
        try:
            data = request.get_json(force=True)  # Force JSON parsing
        except Exception as json_error:
            logging.error(f"JSON parsing error: {json_error}")
            logging.error(f"Raw request data: {request.data}")
            return jsonify({
                'success': False,
                'message': f'Invalid JSON: {str(json_error)}'
            }), 400
        
        # Log the parsed JSON data
        logging.error(f"Parsed JSON data: {data}")
        
        # Validate input with robust error checking
        def safe_get(dictionary, key, default=None):
            """Safely get a value from a dictionary"""
            try:
                return dictionary.get(key, default)
            except Exception as e:
                logging.error(f"Error getting key '{key}': {e}")
                return default
        
        # Extract and validate inputs
        fitness_level = str(safe_get(data, 'fitness_level', 'intermediate')).lower()
        goals = safe_get(data, 'goals', [])
        equipment = safe_get(data, 'equipment', [])
        workout_days = safe_get(data, 'workout_days', [])
        health_conditions = safe_get(data, 'health_conditions', [])
        
        # Log extracted inputs
        logging.error(f"Fitness Level: {fitness_level}")
        logging.error(f"Goals: {goals}")
        logging.error(f"Equipment: {equipment}")
        logging.error(f"Workout Days: {workout_days}")
        logging.error(f"Health Conditions: {health_conditions}")
        
        # Validate workout days
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        workout_days = [day for day in workout_days if day in valid_days]
        
        # Validate health conditions
        valid_conditions = [
            'none', 'hypertension', 'diabetes', 'arthritis', 'obesity', 
            'heart-disease', 'asthma', 'back-pain', 'knee-issues'
        ]
        
        # Comprehensive health condition handling
        if not health_conditions or 'none' in health_conditions:
            health_conditions = ['none']
        else:
            # Ensure only valid conditions are included
            health_conditions = list(set(
                condition for condition in health_conditions 
                if condition in valid_conditions and condition != 'none'
            ))
        
        # Ensure at least one workout day is selected
        if not workout_days:
            logging.error("No workout days selected")
            return jsonify({
                'success': False,
                'message': 'Please select at least one workout day'
            }), 400
        
        EXERCISE_LIBRARY = {
            "warmup": {
                "beginner": [
                    {
                        "name": "Jumping Jacks",
                        "description": "Full-body dynamic warm-up exercise to increase heart rate",
                        "sets": 2,
                        "reps": 20,
                        "muscle_groups": ["full body"],
                        "difficulty": "easy"
                    },
                    {
                        "name": "Arm Circles",
                        "description": "Shoulder mobility exercise to prepare upper body for workout",
                        "sets": 2,
                        "reps": 15,
                        "muscle_groups": ["shoulders", "arms"],
                        "difficulty": "easy"
                    },
                    {
                        "name": "High Knees",
                        "description": "Dynamic movement to activate leg muscles and increase heart rate",
                        "sets": 2,
                        "reps": 20,
                        "muscle_groups": ["legs", "core"],
                        "difficulty": "moderate"
                    }
                ],
                "intermediate": [
                    {
                        "name": "Mountain Climbers",
                        "description": "Dynamic core and cardio exercise to warm up multiple muscle groups",
                        "sets": 3,
                        "reps": 20,
                        "muscle_groups": ["core", "shoulders", "legs"],
                        "difficulty": "moderate"
                    },
                    {
                        "name": "Leg Swings",
                        "description": "Dynamic stretch to improve hip mobility and warm up lower body",
                        "sets": 2,
                        "reps": 15,
                        "muscle_groups": ["hips", "legs"],
                        "difficulty": "easy"
                    }
                ],
                "advanced": [
                    {
                        "name": "Burpees",
                        "description": "High-intensity full-body warm-up exercise",
                        "sets": 3,
                        "reps": 15,
                        "muscle_groups": ["full body"],
                        "difficulty": "hard"
                    }
                ]
            },
            "main_workout": {
                "strength": {
                    "beginner": [
                        {
                            "name": "Bodyweight Squats",
                            "description": "Lower body exercise to build leg strength and stability",
                            "sets": 3,
                            "reps": 12,
                            "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
                            "difficulty": "easy",
                            "equipment": "none"
                        },
                        {
                            "name": "Push-ups",
                            "description": "Classic upper body exercise targeting chest, shoulders, and triceps",
                            "sets": 3,
                            "reps": 10,
                            "muscle_groups": ["chest", "shoulders", "triceps"],
                            "difficulty": "moderate",
                            "equipment": "none"
                        }
                    ],
                    "intermediate": [
                        {
                            "name": "Dumbbell Lunges",
                            "description": "Dynamic leg exercise to improve balance and build lower body strength",
                            "sets": 3,
                            "reps": 12,
                            "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
                            "difficulty": "moderate",
                            "equipment": "dumbbells"
                        },
                        {
                            "name": "Dumbbell Shoulder Press",
                            "description": "Shoulder strengthening exercise to build upper body muscle",
                            "sets": 3,
                            "reps": 10,
                            "muscle_groups": ["shoulders", "triceps"],
                            "difficulty": "moderate",
                            "equipment": "dumbbells"
                        }
                    ]
                },
                "cardio": {
                    "beginner": [
                        {
                            "name": "Jogging in Place",
                            "description": "Low-impact cardio exercise to improve cardiovascular endurance",
                            "duration": "5 minutes",
                            "intensity": "low",
                            "muscle_groups": ["legs", "cardiovascular system"]
                        },
                        {
                            "name": "Jump Rope",
                            "description": "High-energy cardio exercise to burn calories and improve coordination",
                            "sets": 3,
                            "duration": "1 minute",
                            "intensity": "moderate",
                            "muscle_groups": ["full body", "cardiovascular system"]
                        }
                    ]
                }
            },
            "cooldown": {
                "beginner": [
                    {
                        "name": "Standing Quad Stretch",
                        "description": "Stretch to release tension in quadricep muscles",
                        "hold_time": "30 seconds",
                        "muscle_groups": ["quadriceps"]
                    },
                    {
                        "name": "Hamstring Stretch",
                        "description": "Gentle stretch to improve flexibility and reduce muscle tension",
                        "hold_time": "30 seconds",
                        "muscle_groups": ["hamstrings"]
                    }
                ],
                "intermediate": [
                    {
                        "name": "Child's Pose",
                        "description": "Yoga-inspired stretch to relax back and shoulder muscles",
                        "hold_time": "45 seconds",
                        "muscle_groups": ["back", "shoulders"]
                    }
                ]
            }
        }

        # Update base prompt to use exercise library
        base_prompt = f"""\
Generate a comprehensive weekly workout plan using the following exercise library and requirements:

EXERCISE LIBRARY STRUCTURE:
- Warmup Exercises: Categorized by difficulty (beginner/intermediate/advanced)
- Main Workout: Divided into strength and cardio, with difficulty levels
- Cooldown Stretches: Progressive stretching routines

WORKOUT REQUIREMENTS:
- Workout Days: {', '.join(workout_days)}
- Rest Days: {', '.join(day for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] if day not in workout_days)}
- Fitness Level: {fitness_level}
- Goals: {goals}
- Available Equipment: {equipment}

DETAILED INSTRUCTIONS:
1. Select warmup exercises matching user's fitness level
2. Choose main workout exercises based on:
   - Fitness level
   - Available equipment
   - Specific fitness goals
3. Include appropriate cooldown stretches
4. Ensure variety and progressive difficulty

OUTPUT FORMAT (STRICT JSON):
{{
    "metadata": {{
        "difficulty": "{fitness_level}",
        "fitness_goals": {goals},
        "equipment_needed": {equipment}
    }},
    "daily_plan": [
        {{
            "day": "Monday",
            "type": "{"workout" if "Monday" in workout_days else "rest"}",
            "workout_details": {{
                "warmup": {workout_days if "Monday" in workout_days else "null"},
                "main_workout": {workout_days if "Monday" in workout_days else "null"},
                "cooldown": {workout_days if "Monday" in workout_days else "null"}
            }},
            "rest_routine": {workout_days if "Monday" not in workout_days else "null"}
        }}
        # Similar structure for other days
    ]
}}

SPECIFIC GUIDELINES:
- Customize exercises to user's fitness level
- Provide clear, actionable exercise descriptions
- Include sets, reps, and specific muscle group targets
- Ensure safety and gradual progression
"""

        # Prepare the request payload
        payload = {
            "model": os.getenv('OLLAMA_MODEL', 'llama3.2:latest'),
            "prompt": base_prompt,
            "stream": False,
            "format": "json"  # Explicitly request JSON format
        }
        
        # Send request to Ollama
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://10.10.20.9:11434')
        logging.error(f"Ollama Base URL: {base_url}")
        logging.error(f"Ollama Model: {payload['model']}")
        logging.error(f"Prompt sent to Ollama: {payload['prompt']}")
        
        try:
            response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
            
            # Log full response details
            logging.error(f"Ollama Response Status: {response.status_code}")
            logging.error(f"Ollama Response Headers: {response.headers}")
            logging.error(f"Ollama Response Content: {response.text}")
            
            if response.status_code != 200:
                logging.error(f"Ollama API error: {response.text}")
                return jsonify({
                    'success': False,
                    'message': f'Ollama API error: {response.text}'
                }), 500
            
            # Extract the generated text
            response_json = response.json()
            generated_text = response_json.get('response', '')
            
            # Log the generated text for debugging
            logging.error(f"Generated text length: {len(generated_text)}")
            logging.error(f"Generated text (first 500 chars): {generated_text[:500]}")
            
            # Parse the workout plan
            try:
                # Attempt to parse the generated JSON
                workout_plan = json.loads(generated_text)
                
                # Validate the parsed workout plan
                if not isinstance(workout_plan, dict) or 'metadata' not in workout_plan:
                    raise ValueError("Invalid workout plan structure")
                
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Error parsing workout plan: {e}")
                logging.error(f"Raw generated text: {generated_text}")
                return jsonify({
                    'success': False,
                    'message': f'Failed to parse generated workout plan: {str(e)}'
                }), 500
            
            return jsonify({
                'success': True,
                'workout_plan': workout_plan
            })
        
        except requests.exceptions.RequestException as req_error:
            logging.error(f"Request to Ollama failed: {req_error}")
            logging.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': f'Network error: {str(req_error)}'
            }), 500
        
        except Exception as e:
            logging.error(f"Unexpected error in workout generation: {e}")
            logging.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': f'Unexpected error: {str(e)}'
            }), 500
    
    except Exception as outer_error:
        logging.error(f"Outer exception in workout generation: {outer_error}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Unexpected server error: {str(outer_error)}'
        }), 500

@app.route('/register', methods=['POST'])
def register():
    """Handle user registration"""
    data = request.json
    
    # Use new validation method
    if not validate_email(data['email']):
        return jsonify({"error": "Invalid email address"}), 400
    
    # Check if user already exists
    existing_user = User.query.filter(
        (User.username == data['username']) | (User.email == data['email'])
    ).first()
    
    if existing_user:
        return jsonify({"error": "Username or email already exists"}), 409
    
    # Create new user
    new_user = User(
        username=data['username'], 
        email=data['email'],
        age=data.get('age'),
        gender=data.get('gender'),
        fitness_goals=data.get('fitness_goals'),
        health_conditions=data.get('health_conditions')
    )
    new_user.set_password(data['password'])
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    data = request.json
    
    # Find user by username
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        # Store user id in session
        session['user_id'] = user.id
        
        # Check if user has existing workout
        existing_workout = Workout.query.filter_by(user_id=user.id).order_by(Workout.date.desc()).first()
        
        if existing_workout:
            # Redirect to workout page if workout exists
            return jsonify({
                "message": "Login successful", 
                "redirect": "/workout.html", 
                "has_workout": True
            }), 200
        else:
            # Redirect to index to generate new workout
            return jsonify({
                "message": "Login successful", 
                "redirect": "/index.html", 
                "has_workout": False
            }), 200
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/save_workout', methods=['POST'])
def save_workout():
    """Save generated workout for a user"""
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    data = request.json
    
    # Create new workout entry
    new_workout = Workout(
        user_id=session['user_id'],
        difficulty=data.get('difficulty', 'Not specified'),
        total_duration=data.get('total_duration', 'Not specified'),
        fitness_goals=data.get('fitness_goals', 'Not specified')
    )
    new_workout.set_workout_plan(data)
    
    db.session.add(new_workout)
    db.session.commit()
    
    return jsonify({
        "message": "Workout saved successfully", 
        "workout_id": new_workout.id
    }), 201

@app.route('/get_latest_workout', methods=['GET'])
def get_latest_workout():
    """Retrieve the latest workout for the logged-in user"""
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    latest_workout = Workout.query.filter_by(user_id=session['user_id']).order_by(Workout.date.desc()).first()
    
    if not latest_workout:
        return jsonify({"error": "No workouts found"}), 404
    
    workout_data = latest_workout.get_workout_plan()
    return jsonify(workout_data), 200

@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    session.pop('user_id', None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/generate_workout', methods=['POST'])
def generate_workout():
    """Generate workout with optional user context"""
    user_id = session.get('user_id')
    return generate_workout_plan(user_id)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/test_ollama_connection', methods=['GET'])
def test_ollama_connection():
    """
    Comprehensive test of Ollama connection and model availability
    """
    base_url = request.args.get('base_url', 'http://10.10.20.9:11434')
    api_key = request.args.get('api_key', '')
    model_name = 'llama3.2:latest'
    
    logger = logging.getLogger(__name__)
    logger.info(f"Received connection test request. Base URL: {base_url}, Model: {model_name}")
    
    try:
        # Test 1: List available models
        models_url = f"{base_url.rstrip('/')}/api/tags"
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        response = requests.get(models_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        logger.info(f"Available models: {model_names}")
        
        # Test 2: Check if specific model exists
        model_exists = any(model['name'] == model_name for model in models)
        
        if not model_exists:
            logger.warning(f"Model {model_name} not found in available models")
            return jsonify({
                'success': False, 
                'message': f'Model {model_name} not available',
                'available_models': model_names
            }), 404
        
        # Test 3: Attempt a simple generation to verify model works
        test_prompt = "Explain the importance of regular exercise in 3 sentences."
        
        generation_url = f"{base_url.rstrip('/')}/api/generate"
        generation_payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_ctx": 4096,
                "num_predict": -1
            }
        }
        
        gen_response = requests.post(generation_url, json=generation_payload, headers=headers, timeout=300, stream=True)
        gen_response.raise_for_status()
        
        generated_text = ""
        for line in gen_response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode('utf-8'))
                    if 'response' in json_data:
                        generated_text += json_data['response']
                    if json_data.get('done', False):
                        break
                except Exception as parse_error:
                    logging.warning(f"Error parsing stream line: {parse_error}")
        
        generated_text = generated_text.strip()

        logger.info("Model generation test successful")
        
        return jsonify({
            'success': True, 
            'message': 'Successfully connected to Ollama and tested model',
            'available_models': model_names,
            'selected_model': model_name,
            'test_generation': generated_text[:200] + '...' if generated_text else 'No text generated'
        })
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to connect to Ollama at {base_url}',
            'error': str(e)
        }), 500
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {e}")
        return jsonify({
            'success': False,
            'message': 'Error during Ollama API request',
            'error': str(e)
        }), 500
    
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return jsonify({
            'success': False,
            'message': 'Unexpected error during Ollama connection test',
            'error': str(e)
        }), 500

@app.route('/test_model_generation', methods=['GET'])
def test_model_generation():
    """
    Test the workout generation model with a specific prompt
    """
    try:
        base_url = request.args.get('base_url', 'http://10.10.20.9:11434')
        model_name = 'llama3.2:latest'
        
        logger = logging.getLogger(__name__)
        
        # Detailed prompt to encourage direct JSON output
        prompt = """Generate a workout plan in strict JSON format. 
        The JSON should have two main keys: 'metadata' and 'daily_plan'.
        
        Example JSON structure:
        {
            "metadata": {
                "difficulty": "Intermediate",
                "total_duration": "2.5 hours per week",
                "workout_days": ["Monday", "Wednesday", "Friday"],
                "equipment_needed": ["Dumbbells", "Resistance Bands"],
                "fitness_goals": ["Muscle Gain", "Strength"]
            },
            "daily_plan": [
                {
                    "day": "Monday",
                    "type": "workout" or "rest",
                    "warmup": [
                        {"exercise": "Jumping Jacks", "sets": 3, "reps": 30, "rest_period": 30},
                        {"exercise": "Leg Swings", "sets": 3, "reps": 20, "rest_period": 30}
                    ],
                    "main_workout" or "recovery_routine": [
                        {"exercise": "Dumbbell Chest Press", "sets": 4, "reps": 10, "rest_period": 60},
                        {"exercise": "Incline Dumbbell Press", "sets": 3, "reps": 12, "rest_period": 60}
                    ],
                    "cooldown": [
                        {"exercise": "Chest Stretch", "duration": 30},
                        {"exercise": "Arm Circles", "sets": 3, "reps": 15, "rest_period": 30}
                    ]
                }
                // Add other workout days here
            ]
        }

        Specific requirements:
        1. Clearly specify which days of the week the workouts will be performed
        2. Ensure a balanced workout plan with varied exercises
        3. Include warm-up, main workout, and cool-down for each training day
        4. Provide specific exercises with sets, reps, and rest periods
        5. Modify exercises to accommodate any health limitations

        Respond ONLY with a valid JSON that matches the specified output format.
        """
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # Send request to Ollama
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.text}")
            return jsonify({
                "success": False,
                "message": f"Ollama API error: {response.text}"
            }), 500
        
        # Extract the generated text
        generated_text = response.json().get('response', '')
        
        # Parse the workout plan
        workout_plan = parse_workout_plan(generated_text)
        
        return jsonify({
            "success": True,
            "generated_text": generated_text,
            "workout_plan": workout_plan
        })
    
    except Exception as e:
        logger.error(f"Workout generation test failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to generate workout plan",
            "error": str(e)
        }), 500

@app.route('/track_workout', methods=['POST'])
def track_workout():
    """
    Track completed workout and store user progress
    """
    data = request.json
    # TODO: Implement workout tracking and progress storage
    return jsonify({'success': True, 'message': 'Workout tracked successfully'})

@app.route('/log_biometrics', methods=['POST'])
def log_biometrics():
    """Log user's biometric data"""
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    data = request.json
    
    new_biometric_entry = BiometricData(
        user_id=session['user_id'],
        resting_heart_rate=data.get('resting_heart_rate'),
        heart_rate_variability=data.get('heart_rate_variability'),
        sleep_quality=data.get('sleep_quality'),
        stress_level=data.get('stress_level'),
        muscle_soreness=data.get('muscle_soreness'),
        energy_level=data.get('energy_level')
    )
    
    db.session.add(new_biometric_entry)
    db.session.commit()
    
    return jsonify({
        "message": "Biometric data logged successfully",
        "entry_id": new_biometric_entry.id
    }), 201

@app.route('/log_workout_recovery', methods=['POST'])
def log_workout_recovery():
    """Log user's workout recovery and performance"""
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    data = request.json
    
    # Find the most recent workout for this user
    latest_workout = Workout.query.filter_by(
        user_id=session['user_id']
    ).order_by(Workout.date.desc()).first()
    
    new_recovery_log = RecoveryLog(
        user_id=session['user_id'],
        workout_id=latest_workout.id if latest_workout else None,
        perceived_effort=data.get('perceived_effort'),
        recovery_time=data.get('recovery_time'),
        performance_rating=data.get('performance_rating'),
        adaptation_score=data.get('adaptation_score')
    )
    
    db.session.add(new_recovery_log)
    db.session.commit()
    
    return jsonify({
        "message": "Workout recovery logged successfully",
        "log_id": new_recovery_log.id
    }), 201

@app.route('/log_workout', methods=['POST'])
def log_workout():
    """
    Log a completed workout for the current user
    
    Expected JSON payload:
    {
        'user': username,
        'date': ISO formatted date,
        'duration_seconds': total workout duration,
        'workout_plan': full workout plan JSON,
        'completed_exercises': list of completed exercise names
    }
    """
    try:
        data = request.get_json()
        
        # Find the current user
        user = User.query.filter_by(username=data['user']).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Create workout log
        workout_log = WorkoutLog(
            user_id=user.id,
            date=datetime.fromisoformat(data['date']),
            duration_seconds=data['duration_seconds'],
            workout_plan=data['workout_plan'],
            completed_exercises=data['completed_exercises']
        )
        
        # Add and commit to database
        db.session.add(workout_log)
        db.session.commit()
        
        return jsonify({
            "message": "Workout logged successfully", 
            "workout_log_id": workout_log.id
        }), 201
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error logging workout: {str(e)}")
        return jsonify({"error": "Failed to log workout", "details": str(e)}), 500

@app.route('/workout_history', methods=['GET'])
def get_workout_history():
    """
    Retrieve workout history for the current user
    
    Query Parameters:
    - days (optional): Number of days to retrieve history for
    """
    try:
        days = request.args.get('days', default=30, type=int)
        
        # Get current user
        user = User.query.get(session['user_id'])
        
        # Calculate date threshold
        date_threshold = datetime.utcnow() - timedelta(days=days)
        
        # Query workout logs
        workout_logs = WorkoutLog.query.filter(
            WorkoutLog.user_id == user.id,
            WorkoutLog.date >= date_threshold
        ).order_by(WorkoutLog.date.desc()).all()
        
        return jsonify({
            "workout_history": [log.to_dict() for log in workout_logs]
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error retrieving workout history: {str(e)}")
        return jsonify({"error": "Failed to retrieve workout history", "details": str(e)}), 500

def analyze_advanced_personalization(user_id):
    """
    Comprehensive analysis for ultra-personalized workout generation
    
    Args:
        user_id (int): User ID to analyze
    
    Returns:
        dict: Advanced personalization insights
    """
    # Retrieve recent biometric and recovery data
    biometric_data = BiometricData.query.filter_by(
        user_id=user_id
    ).order_by(BiometricData.recorded_at.desc()).limit(7).all()
    
    recovery_logs = RecoveryLog.query.filter_by(
        user_id=user_id
    ).order_by(RecoveryLog.logged_at.desc()).limit(7).all()
    
    # Advanced personalization insights
    personalization_profile = {
        "physiological_readiness": {
            "avg_resting_heart_rate": None,
            "avg_heart_rate_variability": None,
            "avg_sleep_quality": None,
            "avg_stress_level": None,
            "avg_muscle_soreness": None,
            "avg_energy_level": None
        },
        "performance_trends": {
            "avg_perceived_effort": None,
            "avg_recovery_time": None,
            "avg_performance_rating": None,
            "adaptation_trend": None
        },
        "recommendations": []
    }
    
    # Aggregate biometric insights
    if biometric_data:
        bio_metrics = personalization_profile["physiological_readiness"]
        bio_metrics["avg_resting_heart_rate"] = sum(
            data.resting_heart_rate for data in biometric_data if data.resting_heart_rate
        ) / len(biometric_data)
        bio_metrics["avg_heart_rate_variability"] = sum(
            data.heart_rate_variability for data in biometric_data if data.heart_rate_variability
        ) / len(biometric_data)
        bio_metrics["avg_sleep_quality"] = sum(
            data.sleep_quality for data in biometric_data if data.sleep_quality
        ) / len(biometric_data)
        bio_metrics["avg_stress_level"] = sum(
            data.stress_level for data in biometric_data if data.stress_level
        ) / len(biometric_data)
        bio_metrics["avg_muscle_soreness"] = sum(
            data.muscle_soreness for data in biometric_data if data.muscle_soreness
        ) / len(biometric_data)
        bio_metrics["avg_energy_level"] = sum(
            data.energy_level for data in biometric_data if data.energy_level
        ) / len(biometric_data)
    
    # Aggregate performance insights
    if recovery_logs:
        perf_metrics = personalization_profile["performance_trends"]
        perf_metrics["avg_perceived_effort"] = sum(
            log.perceived_effort for log in recovery_logs if log.perceived_effort
        ) / len(recovery_logs)
        perf_metrics["avg_recovery_time"] = sum(
            log.recovery_time for log in recovery_logs if log.recovery_time
        ) / len(recovery_logs)
        perf_metrics["avg_performance_rating"] = sum(
            log.performance_rating for log in recovery_logs if log.performance_rating
        ) / len(recovery_logs)
        
        # Simple adaptation trend calculation
        adaptation_scores = [log.adaptation_score for log in recovery_logs if log.adaptation_score]
        if adaptation_scores:
            perf_metrics["adaptation_trend"] = (
                "Improving" if adaptation_scores[-1] > adaptation_scores[0] else
                "Declining" if adaptation_scores[-1] < adaptation_scores[0] else
                "Stable"
            )
    
    # Generate personalized recommendations
    if personalization_profile["physiological_readiness"]["avg_muscle_soreness"] > 7:
        personalization_profile["recommendations"].append(
            "Consider a recovery-focused workout with low intensity and emphasis on mobility"
        )
    
    if personalization_profile["performance_trends"]["avg_perceived_effort"] > 8:
        personalization_profile["recommendations"].append(
            "Your recent workouts have been very challenging. Consider a deload week."
        )
    
    if personalization_profile["physiological_readiness"]["avg_energy_level"] < 4:
        personalization_profile["recommendations"].append(
            "Low energy levels detected. Focus on nutrition, sleep, and stress management."
        )
    
    return personalization_profile

# Modify generate_workout_plan to incorporate advanced personalization
def generate_workout_plan(user_id=None):
    """
    Generate a personalized workout plan with advanced personalization
    
    Args:
        user_id (int, optional): User ID to generate personalized workout
    
    Returns:
        dict: Generated workout plan
    """
    # ... [existing code] ...
    
    # Retrieve advanced personalization insights
    if user_id:
        try:
            advanced_insights = analyze_advanced_personalization(user_id)
            context['base_prompt'] += f" Advanced user profile: {json.dumps(advanced_insights)}"
        except Exception as e:
            logging.error(f"Error retrieving advanced personalization insights: {e}")
    
    # ... [rest of existing code] ...

if __name__ == '__main__':
    app.run(debug=True, port=5000)
