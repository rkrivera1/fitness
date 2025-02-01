from setuptools import setup, find_packages

setup(
    name='fitness_tracker_app',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask==2.3.2',
        'flask-cors==4.0.0',
        'flask-sqlalchemy==3.0.3',
        'flask-migrate==4.0.4',
        'sqlalchemy==2.0.19',
        'python-dotenv==1.0.0',
        'ollama==0.1.6',
        'requests==2.31.0',
        'pydantic==2.1.1',
        'email-validator==2.0.0',
        'bcrypt==4.0.1',
    ],
    entry_points={
        'console_scripts': [
            'fitness_tracker=app:app.run',
        ],
    },
)
