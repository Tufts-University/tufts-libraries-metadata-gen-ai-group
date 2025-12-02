# 1. Setup and Installation

This section contains detailed instructions for installing Python dependencies, creating a virtual environment, activating it, installing the script using setup.py, and running the program. 
---

## 1.1. Create a virtual environment

```
python3 -m venv venv
```

This creates a new folder named venv containing an isolated Python environment.

## 1.2. Activate the environment

After creating the venv, activate it using the appropriate command for your system.

macOS / Linux
```
source venv/bin/activate
```

Windows (PowerShell)
```
venv\Scripts\activate
```

After activation, your terminal prompt should display:
```
(venv)
```
This indicates the environment is active.

## 1.3. Install Required Packages

Copy code
```
python3 -m pip install --upgrade pip
python3 -m pip install .
```

Running pip install . inside the activated venv installs all dependencies listed in setup.py, including: pandas requests tqdm openpyxl (optional) beautifulsoup4 if added to setup.py

It also registers the script as a Python module so it can be executed from within the venv.

## 2. Running the Program in the Virtual Environment

Once the virtual environment is active and dependencies are installed, you can run the script using:

```
python3 process_genre_fast.py
```