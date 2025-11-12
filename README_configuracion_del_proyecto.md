# Guía de Configuración de Eye Uploader

## Instalación de Python 3.11.7

### Paso 1: Descargar Python

Descarga el instalador de Python versión 3.11.7 desde el sitio oficial:

[Descargar Python para Windows](https://www.python.org/downloads/windows/)

Se descargará un archivo llamado `python-3.11.7-amd64.exe`.

### Paso 2: Instalar Python

Ejecuta el archivo descargado para instalar Python en tu sistema.

### Paso 3: Configurar el Entorno Virtual

Abre la terminal en la raíz del proyecto `EYE_UPLOADER` y navega a la carpeta `functions`:

```powershell
cd functions
```

Busca el path donde está instalado el ejecutable de Python. Por ejemplo:

```
C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe
```

Verifica que la versión de Python sea 3.11.7:

```powershell
C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe --version
```

### Paso 4: Activar el Entorno Virtual e Instalar Dependencias

Activa el entorno virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

Instala las dependencias del proyecto:

```powershell
pip install -r requirements.txt
```

Desactiva el entorno virtual:

```powershell
deactivate
```

Regresa al directorio raíz:

```powershell
cd ..
```

---

## Instalación de Firebase Tools

### Requerimientos Previos

- Cuenta de Google con sesión iniciada en el navegador predeterminado
- NodeJS instalado en el sistema

### Paso 1: Instalar Firebase Tools

Instala `firebase-tools` globalmente usando npm:

```powershell
npm install -g firebase-tools
```

### Paso 2: Iniciar Sesión en Firebase

Inicia sesión en tu cuenta de Firebase:

```powershell
firebase login
```

### Paso 3: Inicializar el Emulador de Firebase

Un emulador es una versión de prueba de los servicios de Firebase que puede ejecutarse localmente.

Inicia el proceso de configuración:

```powershell
firebase init
```

Sigue estos pasos en el asistente de configuración:

1. **You are initializing within an existing Firebase project directory**  
   `Are you ready to proceed? (Y/n):` **Y**

2. Navega con las flechas del teclado hasta la opción **Emulators** y presiona **Espacio** para seleccionarla, luego presiona **Enter**.

3. Navega hasta la opción **Functions Emulator**, presiona **Espacio** para seleccionarla y luego presiona **Enter**.

4. **Would you like to enable the Emulator UI? (Y/n):** **n**

5. **Would you like to download the emulators now? (Y/n):** **Y**

Si todo sale bien, verás el siguiente mensaje de confirmación:

```
+  Wrote configuration info to firebase.json
+  Wrote project information to .firebaserc 
+  Firebase initialization complete!
```

### Paso 4: Verificar Configuración de Firebase

#### Archivo firebase.json

Revisa el archivo `firebase.json` y asegúrate de que la opción `runtime` sea `"python311"`. Si no lo es, edítalo:

**firebase.json**

```json
{
  "functions": [
    {
      "codebase": "default",
      "ignore": [
        "venv",
        ".git",
        "firebase-debug.log",
        "firebase-debug.*.log",
        "*.local"
      ],
      "source": "functions",
      "runtime": "python311"
    }
  ],
  "emulators": {
    "functions": {
      "port": 5001,
      "host": "0.0.0.0"
    },
    "ui": {
      "enabled": false
    },
    "singleProjectMode": true
  }
}
```

#### Archivo .firebaserc

Revisa el archivo `.firebaserc` y asegúrate de que contenga los siguientes valores:

**.firebaserc**

```json
{
  "projects": {
    "default": "eci-ot25"
  }
}
```

### Paso 5: Configurar Credenciales de Firebase

Copia y pega el archivo `firebaseConfig.json` dentro de la carpeta `functions`, al mismo nivel que `main.py`.

**IMPORTANTE:** Asegúrate de proteger este archivo y no publicarlo o exponerlo públicamente, ya que contiene las credenciales necesarias para acceder a los servicios y datos del proyecto en Firebase.

### Paso 6: Iniciar el Emulador de Firebase

Configura el tiempo de espera para la inicialización de funciones:

```powershell
$env:FUNCTIONS_DISCOVERY_TIMEOUT=240
```

Inicia el emulador de funciones:

```powershell
firebase emulators:start --only functions
```

#### Resultado Esperado

Si todo funciona correctamente, deberías ver una salida similar a esta:

```
+  functions: Loaded functions definitions from source: receive_image, receive_pdf, tridimensional_reconstruction.
+  functions[us-central1-receive_image]: http function initialized (http://127.0.0.1:5001/eci-ot25/us-central1/receive_image).
+  functions[us-central1-receive_pdf]: http function initialized (http://127.0.0.1:5001/eci-ot25/us-central1/receive_pdf).
+  functions[us-central1-tridimensional_reconstruction]: http function initialized (http://127.0.0.1:5001/eci-ot25/us-central1/tridimensional_reconstruction).

┌─────────────────────────────────────────────────────────────┐
│   All emulators ready! It is now safe to connect your app. │
└─────────────────────────────────────────────────────────────┘

┌───────────┬──────────────┐
│ Emulator  │ Host:Port    │
├───────────┼──────────────┤
│ Functions │ 0.0.0.0:5001 │
└───────────┴──────────────┘
  Emulator Hub host: 127.0.0.1 port: 4400
  Other reserved ports: 4500

Issues? Report them at https://github.com/firebase/firebase-tools/issues and attach the *-debug.log files.
```

Para detener el servidor, presiona **CTRL+C** en la terminal.

---

## Funciones Disponibles

Las siguientes funciones HTTP estarán disponibles en el emulador:

- `receive_image`: Procesa imágenes subidas (segmentación de melanoma)
- `receive_pdf`: Procesa archivos PDF
- `tridimensional_reconstruction`: Genera reconstrucciones 3D

Cada función estará accesible en `http://127.0.0.1:5001/eci-ot25/us-central1/<nombre-funcion>`.