# Audible Insights: Intelligent Book Recommendations

**Audible Insights** is an intelligent book recommendation system designed to provide personalized audiobook suggestions based on user preferences and listening patterns. By analyzing detailed Audible book data, including ratings, reviews, genres, and listening times, this project aims to create a more tailored audiobook discovery experience.

## Features

- **Personalized Recommendations:** Recommends audiobooks based on user preferences and listening behaviors.
- **Data-Driven Insights:** Analyzes reviews, ratings, and other attributes to suggest high-quality audiobooks.
- **Genre Filtering:** Offers recommendations based on preferred genres and book categories.
- **Advanced Clustering:** Uses machine learning clustering to group similar audiobooks, improving recommendation accuracy.

## Technologies Used

- **Python:** For data analysis and recommendation logic.
- **Pandas & NumPy:** For data manipulation and handling large datasets.
- **Scikit-learn:** For clustering algorithms and machine learning models.
- **Matplotlib/Seaborn:** For data visualization and insights.
- **Jupyter Notebook:** For project development and experimentation.

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
- Required Python libraries (can be installed via `requirements.txt`)

### Installation

1. Clone the repository:
   ``` bash
   git clone https://github.com/your-username/audible-insights.git
   cd audible-insights

2. Install the required libraries:
   ```  bash
   pip install -r requirements.txt


### Running the Project

1. Load the dataset into the environment.
   
2. Execute the recommendation system Jupyter notebook cell by cell.
   - Open the Jupyter Notebook in your browser (e.g., `http://your-ec2-ip:8888`).
   - Navigate to the notebook file and run each cell sequentially to load data, process it, and generate recommendations.

### AWS Hosting Details

To host this project on AWS, you can follow the steps below:

1. Set up an AWS account if you don’t have one already at [AWS Sign Up](https://aws.amazon.com).

2. Launch an EC2 instance:
   - Choose an appropriate instance type (e.g., `t2.micro` for small-scale testing).
   - Use an Ubuntu AMI to install Python, Jupyter Notebook, and the required libraries.
   - Set up security groups to allow SSH (port 22) for access and HTTP (port 80) if you want a web interface.

### Install required dependencies on your EC2 instance:

1. Connect to your EC2 instance via SSH:
   ``` bash
   ssh -i your-key.pem ubuntu@your-ec2-ip


1. Install Python, pip, and other dependencies as mentioned in the "Installation" section.

2. Upload your dataset to the EC2 instance or use an S3 bucket to store it.

3. Launch Jupyter Notebook:

   - Install Jupyter Notebook:
     ``` bash
     pip install notebook
     ```

   - Start the Jupyter server:
     ``` bash
     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
     ```


4. Access the notebook from a web browser by navigating to `http://your-ec2-ip:8888`.

### Contributing
Feel free to fork the project, open issues, and submit pull requests. All contributions are welcome!

### License
This project is licensed under the MIT License - see the LICENSE file for details.




### Deploying on AWS EC2
### 1. Launch an EC2 instance:

- Go to the [AWS Management Console](https://console.aws.amazon.com/).
- Navigate to **EC2** and click on **Launch Instance**.
- Select an appropriate instance type (e.g., `t2.micro` for testing).
- Configure your instance settings, including selecting a **key pair** that you can use for SSH access.
- Ensure that the **Security Group** for your instance allows inbound traffic on:
  - **Port 8501** for Streamlit (default port)
  - **Port 22** for SSH (to access the instance remotely)
  
### 2. SSH into your EC2 instance:

After launching the instance, you can SSH into your EC2 instance using the following command:

``` bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 3. Install necessary dependencies on EC2:

To prepare your EC2 instance, update the system and install the required dependencies:

``` bash
# Update the package list and upgrade existing packages
sudo apt update
sudo apt upgrade -y
```
# Install pip for Python 3
sudo apt install python3-pip -y

# Install Python virtual environment
``` bash
sudo apt install python3-venv -y
```

### 4. Install Python and pip (if not already installed):

If Python and pip are not already installed on your EC2 instance, you can install them using the following command:

``` bash
# Install Python3 pip and other development libraries
sudo apt install python3-pip python3-dev -y
```

### 5. Install Streamlit and other required libraries:

Now, install the necessary Python libraries including Streamlit, pandas, seaborn, matplotlib, and plotly by running the following command:

``` bash
# Install the necessary Python libraries
pip install streamlit pandas seaborn matplotlib plotly
```

### 6. Clone the repository:

Clone your project repository from GitHub to the EC2 instance using the following command:

``` bash
# Clone the repository
git clone https://github.com/your-username/audible-book-recommendations.git
``` 

# Navigate into the project directory
``` bash
cd audible-book-recommendations
```

### 7. Running the Application

#### Activate the virtual environment:

Before running the application, make sure to activate the virtual environment:

``` bash
# Activate the virtual environment
source myenv/bin/activate
```

#### Run the Streamlit app:

To run the Streamlit application on port 8501, use the following command:

``` bash
# Run the Streamlit app
streamlit run app.py --server.port=8501
``` 

#### (Optional) Run the app in the background using `nohup`:

If you want the app to continue running even after you close the terminal, use the following command to run the Streamlit app in the background:

``` bash
# Run Streamlit app in the background
nohup streamlit run app.py --server.port=8501 &
```

### Accessing the Application

1. **Find the EC2 Public IP:**
   - You can find the public IP of your EC2 instance in the **AWS EC2 Dashboard**.

2. **Open the application in your browser:**
   - Use the following URL to access your Streamlit application:

   ``` bash
   http://<EC2_PUBLIC_IP>:8501
   ```

### Accessing the Application

1. **Find the EC2 Public IP:**
   - You can find the public IP of your EC2 instance in the **AWS EC2 Dashboard**.

2. **Open the application in your browser:**
   - Use the following URL to access your Streamlit application:

   ``` bash
   http://51.20.135.71:8501
   ```

## Notes on Hosting

1. **EC2 Instance:**
   - For light workloads, you can use a `t2.micro` EC2 instance. Make sure the instance has enough resources for running the model and Streamlit app.

2. **Security Groups:**
   - Ensure your EC2 instance's security group allows inbound traffic on port `8501` (for Streamlit) and port `22` (for SSH).

3. **Persistence:**
   - Use `nohup` or `screen` to keep the app running in the background, even if you disconnect from the SSH session.

---

## Troubleshooting

1. **Port already in use:**
   - If port `8501` is already in use, you can either:
     - Kill the existing Streamlit process using:
     
       ``` bash
       kill <PID>
       ```
     
     - Or change the port number in the command:
     
       ``` bash
       streamlit run app.py --server.port=8502
       ```

2. **Security Group Issues:**
   - Ensure the security group allows traffic on port `8501` for the Streamlit app and port `22` for SSH.


