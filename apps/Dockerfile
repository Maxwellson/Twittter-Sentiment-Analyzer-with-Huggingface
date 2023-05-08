FROM python:3.10

# Create a virtualenv for dependencies. This isolates these packages from system-level packages.
RUN virtualenv -p python3.10 /env

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Copy the application's requirements.txt and run pip to install all dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
Run pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Add the application source code.
ADD . /app

# The webserver on port 80
Expose 80



# a dependency in requirements.txt.
CMD ['python','Streamlit_app.py']
