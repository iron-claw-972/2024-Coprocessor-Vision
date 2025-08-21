How To Setup A Virtual Environment on the School Computers
==========================================================

To Setup
--------
1. Open a PowerShell terminal. The VSCode builtin one will do.
2. Run `Set-ExecutionPolicy -Scope User Bypass`
3. Ensure you are in the toplevel directory of the repository.
4. Run `py -m venv venv`
5. Run `. .\venv\Scripts\activate.ps1`
6. Ensure the left of the prompt says `(venv)`. If it does not, you have problems.
7. Install dependancies with `py -m pip install -r requirements.txt` If this causes errors, make
sure that pip isn't trying to build stuff from source. Have fun troubleshooting!
8. Run the code with `py main.py`

To Run Code
-----------
Follow steps 2, 5, 6, and 8 from the setup instructions.


How To Setup a Virtual Environment on the Jetson
================================================

To Setup
--------
1. Delete any preexisting virtual environments.
2. Run the setup script.
3. Take a nap.
4. Come back in a few hours / tomorrow.
5. Hope it worked.
6. If it didn't, have fun troubleshooting!

To Run
------
1. Run `source ./venv/bin/activate`
2. Run `python3 main.py`


How To Setup a Virtual Environment on a Linux PC
================================================
1. Run `python3 -m venv venv`
2. Run `source ./venv/bin/activate`
3. Run `python3 -m pip install -r requirements.txt`
4. Running code is the same instructions as for the Jetson.
5. To exit the venv, use `deactivate`

