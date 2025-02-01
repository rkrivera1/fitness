from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import bcrypt
import json
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User account model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # User preferences and profile information
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    fitness_goals = db.Column(db.String(255))
    health_conditions = db.Column(db.String(255))
    
    # Relationships
    workouts = db.relationship('Workout', backref='user', lazy=True)
    workout_stats = db.relationship('WorkoutStatistic', backref='user', lazy=True)
    biometric_data = db.relationship('BiometricData', backref='user', lazy=True)
    recovery_logs = db.relationship('RecoveryLog', backref='user', lazy=True)
    workout_logs = db.relationship('WorkoutLog', backref='user', lazy=True)

    def set_password(self, password):
        """Hash and set the user's password"""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def check_password(self, password):
        """Check if the provided password is correct"""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            self.password_hash.encode('utf-8')
        )

class Workout(db.Model):
    """Generated workout plan model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Workout metadata
    date = db.Column(db.DateTime(timezone=True), server_default=func.now())
    difficulty = db.Column(db.String(50))
    total_duration = db.Column(db.String(50))
    fitness_goals = db.Column(db.String(255))
    
    # Storing workout plan as JSON to handle complex nested structures
    workout_plan = db.Column(db.JSON)
    
    def set_workout_plan(self, plan_dict):
        """Convert workout plan to JSON for storage"""
        self.workout_plan = json.dumps(plan_dict)
    
    def get_workout_plan(self):
        """Retrieve workout plan from JSON"""
        return json.loads(self.workout_plan) if self.workout_plan else None

class WorkoutStatistic(db.Model):
    """Track user's workout performance and progress"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Workout completion tracking
    workout_id = db.Column(db.Integer, db.ForeignKey('workout.id'), nullable=False)
    completed_date = db.Column(db.DateTime(timezone=True), server_default=func.now())
    
    # Performance metrics
    exercises_completed = db.Column(db.Integer, default=0)
    total_exercises = db.Column(db.Integer, default=0)
    calories_burned = db.Column(db.Float)
    
    # Additional tracking
    notes = db.Column(db.Text)
    perceived_difficulty = db.Column(db.Integer)  # 1-10 scale

class BiometricData(db.Model):
    """Track user's physiological data for advanced personalization"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Fitness and health metrics
    resting_heart_rate = db.Column(db.Integer)  # Beats per minute
    heart_rate_variability = db.Column(db.Float)  # HRV score
    sleep_quality = db.Column(db.Float)  # 0-10 scale
    stress_level = db.Column(db.Integer)  # 1-10 scale
    
    # Recovery indicators
    muscle_soreness = db.Column(db.Integer)  # 1-10 scale
    energy_level = db.Column(db.Integer)  # 1-10 scale
    
    # Timestamp
    recorded_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

class RecoveryLog(db.Model):
    """Track user's recovery and adaptation"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Performance and recovery metrics
    workout_id = db.Column(db.Integer, db.ForeignKey('workout.id'))
    perceived_effort = db.Column(db.Integer)  # 1-10 RPE scale
    recovery_time = db.Column(db.Integer)  # Hours needed for full recovery
    
    # Adaptation indicators
    performance_rating = db.Column(db.Float)  # 0-100% performance score
    adaptation_score = db.Column(db.Float)  # Measure of physiological adaptation
    
    # Timestamp
    logged_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

class WorkoutLog(db.Model):
    """
    Model to track individual workout logs and performance
    """
    __tablename__ = 'workout_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    duration_seconds = db.Column(db.Integer, nullable=False)
    workout_plan = db.Column(db.JSON, nullable=False)
    completed_exercises = db.Column(db.JSON, nullable=True)
    
    # Relationship with User model
    user = db.relationship('User', backref=db.backref('workout_logs', lazy=True))

    def to_dict(self):
        """
        Convert WorkoutLog instance to dictionary for JSON serialization
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'date': self.date.isoformat(),
            'duration_seconds': self.duration_seconds,
            'workout_plan': self.workout_plan,
            'completed_exercises': self.completed_exercises
        }

    def __repr__(self):
        return f'<WorkoutLog {self.id} - User {self.user_id} on {self.date}>'

def init_db(app):
    """Initialize the database with the given Flask app"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return db
