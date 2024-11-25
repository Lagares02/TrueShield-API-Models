# API AI Models

## Introducción de la API

Esta API aloja los modelos descargados desde *HuggingFace* para el uso de obtención de palabras claves mediante un prompt y para calcular la inferencia entre un texo y otro. La API utiliza librerias como transformer para el cargue y usos de los modelos

## ¿Cómo está dividida la API?

### Estructura del proyecto

- models/
  - models.py - Archivo que contiene los modelos de HuggingFace
- routes/
  - routes.py - Archivo que contiene las rutas de la API.
- services/
  - logic.py - Archivo que contiene todas la lógica de la API.
- main.py - Archivo principal de la API donde se incluyen las rutas.
- requirements.txt - Archivo con las librerías y paquetes necesarios.

> [!TIP]
> Mantener una estructura modular del proyecto facilita el mantenimiento y la escalabilidad del mismo.

### Componentes

main.py: Contiene la configuración principal de *FastAPI* y la inclusión de las rutas desde el módulo routes.

models: Definición del modelo de datos utilizando *transformers* y se encuentran modelos de NER, POS, NLI y de traduccón (ES-EN y EN-ES).

routes: Definición de las rutas de la API.

services: Definición de las 2 funciones de los modelos utilizados en la API.

### Rutas:
- routes/: La vista del home que solo devuelve un mensaje de "Corriendo exitosamente TrueShield-API-Models!"
- routes/classify: Recibe un prompt mediante una petición *POST* y la manda por parametro a la función que se encuentra en service para obtener las palabras claves y finalmente retorna una lista de keywords en inglés y español
- routes/inference: Obtiene la inferencia de dos textos, mediante la recepción de una hipótesis y una premisa, esto retorna un dicionario que contiene una llave con el tipo de inferencia (si lo afirma, supone o niega) y otra llave que contiene el procentaje de este (toma valores de 0 a 1)
- services/logic.py: Contiene la clase llamada *LTV_Entity_Classifier_Local* que se encarga de hacer la clasificación de las entidades para así agrupar las palabras claves que se retornan, también se encuentra por fuera de esta clase, la función *classify_text_relationship* que realiza el cálculo de la relación con respecto a 2 textos (inferencia)

## Ejecutemos la API

### Iniciamos un entorno virtual (Opcional)

> [!TIP]
> Usar un entorno virtual evita conflictos entre las dependencias de distintos proyectos.

- Abre una terminal y navega al directorio del proyecto.
- Crea el entorno:


python -m venv venv 


- Activa el entorno creado (Para Windows):


.\venv\Scripts\activate


### Instalamos los requerimientos

-   Cuando tengas el entorno virtual activado, puedes instalar las dependencias necesarias:


pip install -r requirements.txt


### Clonar y ejecutar

- Clonamos el repo:


git clone https://github.com/Lagares02/TrueShield-API-Models.git
cd TrueShield-API-Models


- Iniciamos el servidor con tan solo:


py main.py


> [!WARNING]
> Asegúrate de que el puerto 8004 esté libre para evitar conflictos con otras aplicaciones.

- Abre tu navegador y pega este enlace: http://127.0.0.1:8004 para ver la interfaz de usuario.