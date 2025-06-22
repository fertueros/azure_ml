import typer
from pathlib import Path
import pandas as pd
import json
from rich.console import Console # Para imprimir tablas bonitas

# Importar las funciones que ya has creado
from package_ml.modeling.train import train_final_model
from package_ml.modeling.predict import make_predictions, evaluate_model_on_test_set

# Crear la aplicación de línea de comandos
app = typer.Typer()
console = Console() # Para output más legible

# --- Constantes y Rutas ---
DATA_DIR = Path('data/interim')
MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
TARGET = "Calories"

@app.command()
def train(
    model_name: str = typer.Option("final_lgbm_model.joblib", help="Nombre del archivo del modelo a guardar.")
):
    """
    Entrena el modelo final con los mejores parámetros y lo guarda en la carpeta models/.
    """
    console.print("--- [bold green]Iniciando Proceso de Entrenamiento Final[/bold green] ---")
    
    # Cargar configuraciones
    console.print("Cargando datos y configuraciones...")
    df_train = pd.read_parquet(DATA_DIR / 'train_fe.parquet')
    with open(MODELS_DIR / 'final_features.json', 'r') as f:
        final_features = json.load(f)
    with open(MODELS_DIR / 'best_lgbm_params.json', 'r') as f:
        best_params = json.load(f)

    X_train = df_train[final_features]
    y_train = df_train[TARGET]
    
    # Entrenar
    final_model_path = MODELS_DIR / model_name
    train_final_model(X_train, y_train, best_params, final_model_path)
    
    console.print(f"✅ [bold green]Entrenamiento completado. Modelo guardado en {final_model_path}[/bold green]")

@app.command()
def evaluate(
    model_name: str = typer.Option("final_lgbm_model.joblib", help="Nombre del modelo a evaluar.")
):
    """
    Evalúa el modelo final en el conjunto de prueba (test_fe.parquet).
    """
    console.print("--- [bold blue]Iniciando Evaluación del Modelo en Datos No Vistos[/bold blue] ---")
    
    # Cargar datos de prueba
    console.print("Cargando datos de prueba...")
    df_test = pd.read_parquet(DATA_DIR / 'test_fe.parquet')
    with open(MODELS_DIR / 'final_features.json', 'r') as f:
        final_features = json.load(f)
        
    X_test = df_test[final_features]
    y_test = df_test[TARGET]
    
    # Predecir y evaluar
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        console.print(f"❌ [bold red]Error: El modelo {model_path} no existe. Por favor, ejecuta el comando 'train' primero.[/bold red]")
        raise typer.Exit()
        
    predictions = make_predictions(X_test, model_path)
    metrics = evaluate_model_on_test_set(y_test, predictions)
    
    # Imprimir resultados
    from rich.table import Table
    table = Table("Métrica", "Valor")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}")
    
    console.print(table)
    console.print("✅ [bold blue]Evaluación completada.[/bold blue]")

@app.command()
def predict(
    input_path: Path = typer.Argument(..., help="Ruta al archivo .parquet o .csv con los datos para predecir."),
    output_path: Path = typer.Option(None, help="Ruta opcional para guardar las predicciones en un .csv."),
    model_name: str = typer.Option("final_lgbm_model.joblib", help="Nombre del modelo a usar.")
):
    """
    Realiza predicciones en un nuevo archivo de datos.
    """
    console.print(f"--- [bold yellow]Realizando Predicciones en {input_path}[/bold yellow] ---")
    
    # Cargar datos de entrada
    if input_path.suffix == '.parquet':
        input_df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        input_df = pd.read_csv(input_path)
    else:
        console.print("❌ [bold red]Error: Formato de archivo no soportado. Usa .parquet o .csv.[/bold red]")
        raise typer.Exit()
    
    with open(MODELS_DIR / 'final_features.json', 'r') as f:
        final_features = json.load(f)
        
    X_input = input_df[final_features]
    
    # Predecir
    model_path = MODELS_DIR / model_name
    predictions = make_predictions(X_input, model_path)
    
    input_df['predicted_calories'] = predictions
    
    console.print("Predicciones generadas:")
    console.print(input_df.head())
    
    if output_path:
        input_df.to_csv(output_path, index=False)
        console.print(f"✅ [bold yellow]Predicciones guardadas en {output_path}[/bold yellow]")
        
if __name__ == "__main__":
    app()