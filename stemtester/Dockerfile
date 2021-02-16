# Python image to use.
FROM python:3.8


# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt . 

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

EXPOSE 8503 

# Run streamlit
RUN ["chmod", "+x", "start_streamlit.sh"]

CMD ./start_streamlit.sh

#docker run --rm -d -p 5000:5000/tcp -p 8501:8501/tcp chatbottbtb:latest