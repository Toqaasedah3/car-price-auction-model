# Auction Car Price Prediction

## 📄 Description
This project predicts car prices in auctions using machine learning models. The data is processed and cleaned to ensure accuracy, and the project utilizes Data Version Control (DVC) for managing datasets and versioning.

## 🖂 Project Structure
```
car-price-auction-model/
├── .gitignore          # Git ignore file
├── README.md           # Project documentation
├── car.ipynb           # Jupyter Notebook with the model and analysis
├── car_prices.csv.dvc  # DVC-tracked dataset reference
└── dvc.lock            # DVC lock file for dataset versioning
```

## 🚀 Features
- Predict car prices based on auction data.
- Clean and preprocess datasets.
- Utilize DVC to manage datasets and version control.

## 📦 Prerequisites
Make sure the following tools are installed on your machine:
1. Python (>=3.8)
2. Jupyter Notebook
3. Git
4. DVC

Install project dependencies:
```bash
pip install -r requirements.txt
```

## 📥 How to Download Data
1. Clone the repository:
   ```bash
   git clone https://github.com/Toqaasedah3/car-price-auction-model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd car-price-auction-model
   ```
3. Pull the dataset using DVC:
   ```bash
   dvc pull
   ```

## 🛠️ How to Run the Project
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook car.ipynb
   ```
2. Follow the notebook steps to load the data, clean it, and build the machine learning model.

## 🕰 Dataset
- The dataset contains auction data for car prices, including features like mileage, year of manufacture, and condition.
- **Dataset Source**: 
  - Kaggle: [Used Car Auction Prices Dataset](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices/data)

## 🧰 Tools and Technologies
- **Python**: Programming language
- **Jupyter Notebook**: For interactive analysis
- **DVC**: Dataset versioning
- **Google Drive**: Storage backend for DVC

## 📚 Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Matplotlib**: For basic data visualization and plotting.
- **Seaborn**: For advanced and aesthetic statistical visualizations.
- **Plotly**: For interactive and dynamic visualizations:
  - `plotly.express`: Simplified interface for Plotly visualizations.
  - `plotly.graph_objects`: For more customizable plots.
  - `plotly.offline`: Offline visualization support.
  - `plotly.io`: Input-output functionalities for Plotly.
- **Warnings**: To suppress unnecessary warnings during execution.

## ✨ Future Improvements
- Add more advanced machine learning models.
- Visualize predictions with an interactive dashboard.
- Optimize the model for better accuracy.

## 📧 Contact
For any questions or collaboration inquiries, feel free to reach out to:
- **GitHub**: [Toqaasedah3](https://github.com/Toqaasedah3)
- **Collaborator**: [Bilal Anbosi](https://github.com/bilal-anabosi)
