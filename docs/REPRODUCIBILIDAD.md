# Guía de Reproducibilidad del Proyecto

Esta guía describe **paso a paso** cómo aprovisionar la infraestructura, instalar las herramientas necesarias y clonar este repositorio para que puedas replicar los análisis desde cero.

---

## Requisitos previos

| Recurso                     | Detalles                                                                                     |
|-----------------------------|----------------------------------------------------------------------------------------------|
| **Suscripción de Azure**    | Permisos para crear máquinas virtuales y asignar direcciones IP públicas.                    |
| **Cuenta de GitHub**        | Con capacidad de generar/gestionar claves SSH.                                               |
| **Cliente SSH**             | macOS/Linux: terminal nativa · Windows: [Powershell 7](https://aka.ms/powershell) o WSL 2.   |
| **Conocimientos básicos**   | Uso de la terminal y comandos `git`.                                                         |

> **Tip:** Todas las instrucciones asumen un sistema operativo **Ubuntu 24.04 LTS** tanto en la VM como en tu máquina local. Las instrucciones se realizaron sobre **bash** en una **MacBook Pro**.

---

## 1 · Aprovisionar la máquina virtual en Azure

1. Inicia sesión en el [portal de Azure](https://portal.azure.com/) y crea una **VM** con:
   - **Imagen:** *Ubuntu Server 24.04 LTS*
   - **Tamaño recomendado:** *Standard B2s* (2 vCPU | 4 GiB)
   - **Autenticación:** *SSH public key* (sube tu clave pública o créala en Azure Cloud Shell)
2. Abre el puerto `22` en el **grupo de seguridad de red** (NSG) para permitir conexiones SSH.
3. Anota la **IP pública** que Azure le asignó a la VM; la utilizarás más adelante.

---

## 2 · Conectar vía SSH y preparar la VM

### 2.1 Configurar la clave SSH

```bash
# Ajustar permisos de la clave privada (ejecutar en tu máquina local)
chmod 600 ~/.ssh/VM_key.pem
```

Opcional: define un *host alias* en tu archivo `~/.ssh/config` para simplificar la conexión:

```bash
Host azure-ml
    HostName <IP_PÚBLICA_AZURE>
    User azureuser
    IdentityFile ~/.ssh/VM_key.pem
```

Conecta:

```bash
ssh azure-ml
```

### 2.2 Actualizar paquetes y dependencias básicas

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common git build-essential
```

### 2.3 Instalar Python 3.11 y pip

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
```

### 2.4. Configuración de GitHub en la VM

1.  **Generar una Nueva Clave SSH:**
    *   Dentro de la VM, genera una nueva clave SSH para vincularla con tu cuenta de GitHub.
    ```bash
    ssh-keygen -t ed25519 -C "tu_correo@ejemplo.com"
    ```
    *   Presiona `Enter` en todas las preguntas para aceptar las opciones por defecto.

2.  **Añadir la Clave a GitHub:**
    *   Muestra la clave pública y cópiala en tu portapapeles.
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    *   Ve a tu cuenta de GitHub > **Settings** > **SSH and GPG keys** y añade la nueva clave pública.

3.  **Verificar la Conexión:**
    *   Confirma que la conexión con GitHub se ha establecido correctamente.
    ```bash
    ssh -T git@github.com # Deberías ver un mensaje de bienvenida
    ```

---

## 3 · Instalar herramientas con `pipx`

`pipx` permite aislar paquetes globales sin contaminar tu entorno Python de sistema.

```bash
sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc # recarga la sesión para habilitar pipx

# Herramientas de este proyecto
pipx install cookiecutter-data-science
pipx install uv
```

---

## 4 · Generar la plantilla del proyecto con *Cookiecutter Data Science*

> **Nota:** El comando `ccds` es un *wrapper* de Cookiecutter incluido en `cookiecutter-data-science`.

```bash
ccds
```

Responde a las preguntas del asistente interactivo tal como se muestra (puedes cambiar valores a tu gusto):

```
project_name (project_name): Regresión de Calorías
repo_name (regresion_de_calorias): azure_ml
module_name (regresion_de_calorias): package_ml
author_name (...): fertueros
description (...): Proyecto para predecir calorías quemadas
python_version_number (3.10): 3.11
Select dataset_storage (1/4): 1  # None
Select environment_manager (1/5): 4  # uv
Select dependency_file (1/4): 2  # pyproject.toml
Select pydata_packages (1/2): 1  # None
Select testing_framework (1/3): 1  # None
Select linting_and_formatting (1/2): 1  # ruff
Select open_source_license (1/3): 2  # MIT
Select docs (1/2): 2  # None
Select include_code_scaffold (1/2): 1  # Yes
```

---

## 5 · Crear y activar el entorno `uv`

```bash
cd azure_ml           # dir. generada por Cookiecutter
uv venv az_ml         # crea entorno virtual
source az_ml/bin/activate
```

Instala las dependencias principales:

```bash
uv add numpy pandas scikit-learn matplotlib seaborn jupyterlab lightgbm xgboost
```

> **⚠️ Exclusión del entorno**  
> Añade `az_ml/` bajo la sección **# Environments** de tu `.gitignore` para evitar subir el entorno virtual.

---

## 6 · Inicializar repositorio Git y publicar en GitHub

```bash
# Dentro de azure_ml/
git init -b main

git remote add origin git@github.com:fertueros/azure_ml.git

git add .
git commit -m "feat: initial project structure from cookiecutter"

git push -u origin main
```

---

## 7 · Próximos pasos

1. **Abrir JupyterLab** para explorar los notebooks:
   ```bash
   uv run jupyter lab --ip 0.0.0.0 --no-browser
   ```