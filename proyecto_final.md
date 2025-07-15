# Informe Proyecto IA I
# Music Track Mixer

## Introducción

El presente proyecto tiene como principal objetivo evaluar el rendimiento de un algoritmo genético para generar piezas musicales, más específicamente, para crear nuevas composiciones musicales partiendo de un determinado número de pistas de entrada, y que estas nuevas creaciones evoquen de alguna forma aspectos musicales de las piezas originales.

La aplicación de algoritmos de machine learning al campo de la música ha resultado de interés a lo largo de mucho tiempo, debido en primer lugar a la dificultad de lograr capturar todos los patrones y sutilezas que residen en una composición musical, ya sea con el fin de crear nuevas composiciones o de clasificar grupos de canciones ya existentes según género, intérprete, tipo de instrumentos, etc., y por otro lado debido también a lo subjetivo que puede resultar valorar las cualidades musicales de una determinada pieza. Actualmente, y de acuerdo a lo investigado, la gran mayoría de soluciones aplicadas a este campo se apoyan principalmente en la utilización de algoritmos de aprendizaje profundo para extraer características y patrones musicales, obteniendo resultados destacables. Sin embargo, dado que este tipo de algoritmos conllevan un costo computacional muy alto, resulta interesante considerar la posibilidad de aplicar otro tipo de algoritmos para este fin, y es por eso que para este trabajo en particular se propone entonces la utlización de un algoritmo genético.

En las siguientes secciones se amplía acerca de los fundamentos teóricos (tanto musicales como algorítmicos) necesarios para entender el funcionamiento del algoritmo y sus limitaciones, se presenta el diseño de los experimentos llevados adelante para evaluar su rendimiento junto con los resultados obtenidos y finalmente se analizan estos resultados y se comenta respecto de posibles mejoras o modificaciones.

## Marco Teórico

### 1. ALGUNOS FUNDAMENTOS DE TEORÍA MUSICAL

En el desarrollo de una pieza musical entran en juego diversos conceptos, entre los cuales podemos mencionar:

### 1.1 Elementos Básicos de la Música

**Melodía**: se define como la secuencia de notas musicales organizadas en el tiempo que forman una línea musical reconocible. Es la parte "cantable" de una pieza musical y generalmente constituye el elemento que más se recuerda. Una melodía se caracteriza por su contorno (dirección ascendente o descendente), intervalos entre notas y ritmo.

**Armonía**: hace referencia a aquellas notas que se ejecutan de forma simultánea y que dan forma a los acordes y las progresiones armónicas. Proporciona el contexto tonal y el soporte estructural para la melodía.

**Ritmo**: constituye la organización temporal de los sonidos musicales, incluyendo la duración de las notas, patrones rítmicos y el pulso.

**Timbre**: se define como la cualidad del sonido que permite distinguir entre diferentes instrumentos o voces, incluso cuando tocan la misma nota.

### 1.2 Representación de Alturas Musicales

En el sistema musical occidental, las alturas se organizan en octavas, donde cada octava contiene 7 tonos y 12 semitonos. La nomenclatura de las notas en este sistema es la siguiente:

- **Notación tradicional**: Do, Re, Mi, Fa, Sol, La, Si
- **Notación anglosajona**: C, D, E, F, G, A, B
- **Representación numérica**: cada semitono corresponde a un número entero

### 1.3 Intervalos y Tonos Relativos

Los **intervalos** son las distancias entre dos notas musicales, medidas en semitonos. Los tonos relativos representan las diferencias de altura entre notas consecutivas. Por ejemplo, una secuencia melódica como C-D-E-F se puede representar como [+2, +2, +1] en semitonos, donde cada número indica la diferencia respecto a la nota anterior. Utilizar entonces tonos relativos en lugar de notas absolutas presenta las siguientes ventajas:

**Invariancia tonal**: una melodía mantiene su identidad independientemente de la tonalidad en que se toque.

**Reducción de espacio de búsqueda**: el algoritmo puede enfocarse en las relaciones melódicas sin preocuparse por la tonalidad absoluta.

**Facilidad de transposición**: las melodías pueden transponerse fácilmente sumando una constante a todos los valores tonales.

### 2. PROTOCOLO MIDI

El protocolo **MIDI** (Musical Instrument Digital Interface) permite comunicar instrumentos musicales electrónicos con computadoras y otros equipos similares. Fue desarrollado en la década de 1980 y se ha convertido actualmente en el estándar de facto para la representación digital de música, debido principalmente las siguientes características:

- **Eficiencia de almacenamiento**: MIDI no almacena audio digitalizado, sino instrucciones sobre cómo tocar música. Esto resulta en archivos extremadamente pequeños comparados con formatos de audio.

- **Editabilidad**: los datos MIDI pueden modificarse fácilmente, permitiendo cambiar notas, instrumentos, tempo y otros parámetros sin degradación de calidad.

- **Separación de contenido**: se separa la información musical (qué tocar) de la síntesis de sonido (cómo suena), facilitando la manipulación algorítmica.

- **Precisión temporal**: se manejan tiempos con alta precisión, permitiendo representar ritmos complejos y matices temporales.

La información musical contenida en un archivo MIDI se representa mediante una serie de "mensajes" como los siguientes:

**Note On/Off**: especifica cuándo comienza y termina una nota, incluyendo:
- Pitch (altura): número de 0 a 127, donde 60 representa el Do (o C) central
- Velocity (velocidad): intensidad de ataque (0-127)
- Channel (canal): permite hasta 16 instrumentos simultáneos

**Program Change**: selecciona el instrumento o timbre a utilizar

**Control Change**: modifica parámetros como volumen, modulación, etc.

**Timing**: provee información temporal precisa sobre cuándo ocurren los eventos

Para el proyecto en particular, se utiliza la librería `pretty-midi` que abstrae muchos de estos parámentros, facilitando su manipulación.



## Diseño Experimental


## Análisis de resultados


## Conclusión


## Bibliografía

