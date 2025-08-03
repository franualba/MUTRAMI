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

- **Melodía**: se define como la secuencia de notas musicales organizadas en el tiempo que forman una línea musical reconocible. Es la parte "cantable" de una pieza musical y generalmente constituye el elemento que más se recuerda. Una melodía se caracteriza por su contorno (dirección ascendente o descendente), intervalos entre notas y ritmo.

- **Armonía**: hace referencia a aquellas notas que se ejecutan de forma simultánea y que dan forma a los acordes y las progresiones armónicas. Proporciona el contexto tonal y el soporte estructural para la melodía.

- **Ritmo**: constituye la organización temporal de los sonidos musicales, incluyendo la duración de las notas, patrones rítmicos y el pulso.

- **Timbre**: se define como la cualidad del sonido que permite distinguir entre diferentes instrumentos o voces, incluso cuando tocan la misma nota.

### 1.2 Representación de Alturas Musicales

En el sistema musical occidental, las alturas se organizan en octavas, donde cada octava contiene 7 tonos y 12 semitonos. La nomenclatura de las notas en este sistema es la siguiente:

- **Notación tradicional**: Do, Re, Mi, Fa, Sol, La, Si
- **Notación anglosajona**: C, D, E, F, G, A, B
- **Representación numérica**: cada semitono corresponde a un número entero

### 1.3 Intervalos y Tonos Relativos

Los **intervalos** son las distancias entre dos notas musicales, medidas en semitonos. Los tonos relativos representan las diferencias de altura entre notas consecutivas. Por ejemplo, una secuencia melódica como C-D-E-F se puede representar como [+2, +2, +1] en semitonos, donde cada número indica la diferencia respecto a la nota anterior. Utilizar entonces tonos relativos en lugar de notas absolutas presenta las siguientes ventajas:

- **Invariancia tonal**: una melodía mantiene su identidad independientemente de la tonalidad en que se toque.
- **Reducción de espacio de búsqueda**: el algoritmo puede enfocarse en las relaciones melódicas sin preocuparse por la tonalidad absoluta.
- **Facilidad de transposición**: las melodías pueden transponerse fácilmente sumando una constante a todos los valores tonales.

### 2. PROTOCOLO MIDI

El protocolo **MIDI** (Musical Instrument Digital Interface) permite comunicar instrumentos musicales electrónicos con computadoras y otros equipos similares. Fue desarrollado en la década de 1980 y se ha convertido actualmente en el estándar de facto para la representación digital de música, debido principalmente las siguientes características:

- **Eficiencia de almacenamiento**: MIDI no almacena audio digitalizado, sino instrucciones sobre cómo tocar música. Esto resulta en archivos extremadamente pequeños comparados con formatos de audio.
- **Editabilidad**: los datos MIDI pueden modificarse fácilmente, permitiendo cambiar notas, instrumentos, tempo y otros parámetros sin degradación de calidad.
- **Separación de contenido**: se separa la información musical (qué tocar) de la síntesis de sonido (cómo suena), facilitando la manipulación algorítmica.
- **Precisión temporal**: se manejan tiempos con alta precisión, permitiendo representar ritmos complejos y matices temporales.

La información musical contenida en un archivo MIDI se representa mediante una serie de "mensajes" como los siguientes:

- **Note On/Off**: especifica cuándo comienza y termina una nota, incluyendo:
    - Pitch (altura): número de 0 a 127, donde 60 representa el Do (o C) central
    - Velocity (velocidad): intensidad de ataque (0-127)
    - Channel (canal): permite hasta 16 instrumentos simultáneos
- **Program Change**: selecciona el instrumento o timbre a utilizar
- **Control Change**: modifica parámetros como volumen, modulación, etc.
- **Timing**: provee información temporal precisa sobre cuándo ocurren los eventos

Para el proyecto en particular, se utiliza la librería `pretty-midi` que abstrae muchos de estos parámentros, facilitando su manipulación.

### 3. ALGORITMOS GENÉTICOS

En el ámbito del machine learning, se conoce como "algoritmos genéticos" al tipo de algoritmos evolutivos que emplean técnicas de **búsqueda local** para encontrar soluciones a problemas de **optimización** y aprendizaje, inspirados en el proceso evolutivo de la **naturaleza**.

Haciendo una analogía con el concepto biológico, entre los principales elementos de estos algoritmos encontramos entonces una **población** compuesta por un conjunto de **individuos** donde cada uno representa una posible solución codificada del problema. Estos últimos normalmente son alterados o modificados en distintas proporciones y siguiendo diferentes criterios, por medio de lo que se denomina "**operadores genéticos**", de los cuales podemos mencionar los de selección, entrecruzamiento o recombinación, mutación, fisión, elisión, entre otros. Como resultado de aplicar estos operadores a los individuos de una población se genera una nueva población, evolucionada, para la cual vuelve a repetirse el proceso de aplicación de operadores. Cada una de estas "nuevas poblaciones" recibe el nombre de **generación**, y la pieza clave encargada de evaluar la idoneidad de cada individuo entre generación y generación y guiar así todo el proceso evolutivo se denomina **función de aptitud o de fitness**. De esta manera, la evolución puede finalizar atendiendo a diversos criterios, siendo el principal el hecho de haber obtenido un valor deseado de aptitud, pero incluyendo también el haber alcanzado un número determinado de iteraciones o haber excedido un valor particular en el tiempo de ejecución del algoritmo.

En líneas generales, podemos decir que el funcionamiento básico de este tipo de algoritmos se resume de la siguiente manera:

```
1. Se inicializa una población comunmente aleatoria
2. Se calcula el valor de fitness de cada individuo
3. Mientras no se cumpla el criterio de detención:
   a. Se seleccionan individuos para reproducción
   b. A una parte de estos se les aplica recombinación
   c. A otra parte mutación
   d. Se calcula el valor de fitness de los nuevos individuos
   e. Se seleccionan aquellos individuos que integrarán la próxima generación
4. Se retorna el mejor individuo 
```

### 4. NORMALIZED COMPRESSION DISTANCE

Como se mencionó previamente en la introducción, la parte tal vez más difícil de aplicar un análisis algorítmico al campo de la música resulta ser la de cómo definir lo que pueda llegar a ser una composición musical idónea o "más apta" que otra, ya que esto se presta a depender del punto de vista desde el cual una pieza musical pueda ser juzgada por un eventual oyente.

Para intentar construir una función de aptitud objetiva a fin de evitar entonces este grado de subjetividad inherente al tema en estudio, algunos investigadores optaron por experimentar con un concepto perteneciente al ámbito de la Teoría de la Información denominado **Normalized Information Distance** (NID), cuya fórmula se define a continuación:

$$\text{NID}(x,y) = \frac{\max\{K(x|y), K(y|x)\}}{\max\{K(x), K(y)\}}$$

donde $K(x|y)$ es la complejidad condicional de Kolmogorov de la cadena $x$ dada la cadena $y$, y cuyo valor es la longitud del programa más corto (para alguna máquina universal) el cual al proporcionarle como entrada la cadena $y$ devuelve la cadena $x$. 

Desafortunadamente, y como se menciona en [2], acudiendo al problema de la detención de máquinas de Turing puede demostrarse que tanto las complejidades condicionales como incondicionales presentes en la fórmula anterior resultan ser funciones no computables, por lo cual en la práctica se utilizan aproximaciones a la complejidad de Kolmogorov empleando algoritmos de compresión ya existentes y computables (como gzip, bzip2, lz4, etc.) que dan lugar a la métrica llamada **Normalized Compression Distance** (NCD):

$$\text{NCD}(x,y) = \frac{C(xy) - \min\{C(x), C(y)\}}{\max\{C(x), C(y)\}}$$

donde:
- $C$ es un algoritmo de compresión determinado
- $C(x)$ es el tamaño comprimido de la cadena $x$ usando C
- $C(y)$ es el tamaño comprimido de la cadena $y$ usando C
- $C(xy)$ es el tamaño comprimido de la cadena concatenada $xy$ usando C

Si bien esta métrica no logra los resultados teóricamente óptimos de su versión no computable, se ha demostrado que en la práctica, y en algunas ocasiones combinándola con otras técnicas (como KNN, ver [2]), los resultados obtenidos al clasificar piezas musicales por género (además también de los resultados obtenidos con tareas de clustering más allá de la música) son muy satisfactorios, motivo por el cual se espera poder replicar estos resultados al utilizar la NCD como parte de la función de fitness de un algoritmo genético.

Considerando que aplicamos esta métrica entonces para evaluar el grado de similitud entre dos secuencias de tonos relativos $x$ e $y$, el valor obtenido puede interpretarse de la siguiente manera:

- **NCD ≈ 0**: Las secuencias son muy similares
- **NCD ≈ 1**: Las secuencias son completamente diferentes
- **NCD > 1**: Puede ocurrir debido a imperfecciones del algoritmo de compresión

Para su uso en la función de fitness, se opta por invertir la relación anterior, de forma que, a mayor valor de fitness (es decir, un valor de dicha función más cercano a uno), indique mayor similitud entre las secuencias y por lo tanto mayor valor de aptitud.

## Diseño Experimental

Para poner a prueba el rendimiento del algoritmo planteado se establecieron como punto de partida los siguientes lineamientos:

1. Sólo se trabaja con la melodía de los archivos musicales, dejando de lado aspectos como la armonía o el ritmo. Esto implica que, para un archivo MIDI en particular obtenido de internet, se debe inspeccionar el número de pistas incluídas en el mismo, quedándonos sólo con la parte melódica en caso de encontrar más de una pista.
2. Se omite alterar la duración de cada nota, trabajando únicamente con los tonos. Esto se debe a que, de acuerdo a lo investigado (ver [2]), si se modifica la duración de cada nota en una melodía sin afectar sus tonos, sigue siendo posible reconocer la melodía original (en otras palabras, la duración de cada nota no altera de forma sustancial la esencia de una melodía). Sin embargo esto no ocurre con el caso opuesto, es decir, al alterar los tonos de cada nota de una melodía manteniendo sus duraciones, se ha comprobado que la melodía deja de ser reconocible.
3. Cada individuo de la población consiste en una secuencia de diferencias tonales, es decir, se trabaja con tonos relativos y no absolutos.
4. Para guiar el proceso evolutivo, se toman como guía dos secuencias melódicas, de la misma longitud que los individios de la población. La función de fitness calcula entonces el valor de aptitud de cada individuo partiendo del valor de la NCD entre la secuencia tonal del individuo y la secuencia de la melodía guía, para cada secuencia guía, y ponderando estas distancias en partes iguales según el número de secuencias guía (en este caso, se utilizan 2 secuencias guía, por lo que cada valor de la NCD se multiplica por 0.5 para obtener el valor de aptitud final de un individuo).
5. La mutación se realiza eligiendo en forma aleatoria un valor puntual de la secuencia de tonos relativos del individuo, y adicionándole un valor también aleatorio en el rango [-2, 2].
6. La longitud de cada individuo es de 50 tonos relativos, y cada tono se inicializa en forma aleatoria con un valor en el rango [-20, 20].
7. Se utiliza un tamaño de población de 500 individuos.
8. Se ejecuta 30 veces cada proceso evolutivo con el objetivo de obtener resultados más representativos del rendimiento de una estrategia. Por ejemplo, si una estrategia se ejecuta a lo largo de 100 generaciones, se repite 30 veces cada proceso de 100 generaciones y se agregan los resultados de cada generación al final para su posterior evaluación.

La estrategia de evolución principal adoptada consiste en:
1. Inicializar una población en forma aleatoria.
2. Calcular el valor de fitness de cada individuo y ordenar la población en forma descendente de acuerdo a este valor.
3. Eliminar el 25% menos apto de individuos de la población.
4. Aplicar una estrategia de recombinación al 25% de los individuos más aptos, y agregar el resultado al resto de la población para sustituir el 25% eliminado en el paso anterior.
5. Aplicar mutación a todos los individuos.
6. Repetir desde el paso 2 hasta llegar al número de generaciones deseado.

Con el propósito de investigar la mejor relación entre exploración y explotación del algoritmo, se experimenta con distintas estrategias de recombinación, identificadas de la siguiente forma:

1. **Estrategia 1**: se utiliza siempre recombinación de un solo punto
2. **Estrategia 2**: se utiliza siempre recombinación de dos puntos
3. **Estrategia 3**: del total de generaciones a ejecutar, el primer 20% utiliza recombinación doble y el 80% restante recombinación simple 
4. **Estrategia 4**: se utiliza siempre recombinación de tres puntos
5. **Estrategia 5**: se utiliza siempre recombinación de cuatro puntos
6. **Estrategia 6**: las primeras 200 generaciones utilizan solo recombinación doble, entre las generaciones 201 y 500 se emplea solo recombinación triple, y de la generación 501 en adelante se aplica únicamente recombinación simple.
7. **Estrategia 7**: solución aleatoria (no emplea recombinación ni mutación, pero sí elitismo) 

A continuación se presentan algunas figuras con los resultados obtenidos:

<div align="center">

![](./plots/plot3_fixed_indsize50_popsize500_gens100_single.png)  
<b>Figura 1.</b> Ejecución simple de 100 generaciones, cada una de 500 individuos y cada individuo con una longitud de 50 tonos relativos
</div>

<div align="center">

![](./plots/plot3_fixed_indsize50_popsize500_gens100_single_runs30.png)  
<b>Figura 2.</b> Mismos parámentros de ejecución que los de la Figura 1, pero repitiendo la ejecución de cada proceso evolutivo 30 veces (es decir, 30 ejecuciones de 100 generaciones cada una)
</div>

<div align="center">

![](./plots/plot4_fixed_indsize50_popsize500_gens1000_single_runs30.png)  
<b>Figura 3.</b> Resultado de ejecutar 30 procesos de 1000 generaciones cada uno, donde cada generación consta de 500 individuos de 50 tonos relativos cada uno
</div>

<div align="center">

![](./plots/plot_indsize50_popsize500_gens1000_multi_strategy_runs30.png)  
<b>Figura 4.</b> Diagrama de líneas de tendencia comparando el rendimiento de las distintas estrategias de recombinación, para 30 procesos de 1000 generaciones cada uno, donde cada generación consta de 500 individuos de 50 tonos relativos cada uno
</div>

<div align="center">

![](./plots/plot_indsize50_popsize500_gens1000_multi_strategy_runs30_zoomed_0.png)  
<b>Figura 5.</b> Acercamiento a la parte final del proceso evolutivo representado en la Figura 4 (incluye desde la generación número 700 a la número 1000)
</div>

<div align="center">

![](./plots/plot_indsize50_popsize500_gens1000_multi_strategy_runs30_zoomed_1.png)  
<b>Figura 6:</b> Acercamiento a la parte inicial del proceso evolutivo representado en la Figura 4 (incluye desde la generación número 50 a la número 400)
</div>

<div align="center">

![](./plots/plot_indsize50_popsize500_gens100_multi_strategy_runs30_elitism_random.png)  
<b>Figura 7.</b> Muestra el rendimiento de cada estrategia como en la Figura 4, pero esta vez empleando elitismo (me quedo con las 2 mejores soluciones de cada generación) e incluyendo también a modo de comparación el rendimiento de una solución aleatoria (la cual no emplea ni recombinación ni mutación, pero si elitismo). Esta prueba consistió en 30 procesos de 100 generaciones cada uno (no 1000), donde cada generación consta de 500 individuos de 50 tonos relativos cada uno
</div>

Y la siguiente es una tabla que permite visualizar algunos tiempos de ejecución de cada estrategia junto con el mejor valor de fitness obtenido:

<div align="center">

| Estrategia | Descripción | Best Fitness | Tiempo Total | Tiempo Promedio | Más Rápida | Más Lenta | Desv. Est. |
|:------------:|:----------------------------:|:--------------:|:-----------------:|:------------:|:-----------:|:------------:|:------------:|
| 1          | Recombinación simple       | 0.2746  | 234.75s      | 7.82s            | 7.37s      | 8.39s     | 0.21s      |
| 2          | Recombinación doble        | 0.2908  | 256.74s      | 8.56s            | 7.19s      | 13.97s    | 1.51s      |
| 3          | Recombinación mixta        | 0.2810  | 232.86s      | 7.76s            | 7.28s      | 8.29s     | 0.22s      |
| 4          | Solución aleatoria         | 0.1972  | 204.90s      | 6.83s            | 6.67s      | 7.25s     | 0.12s      |

<b>Tabla 1</b>

</div>

<div align="center">

![](./plots/plot_indsize75_popsize500_gens1000_multi_strategy_runs30_elitism_random.png)  
<b>Figura 8.</b> Se agregan al experimento tres nuevas estrategias de recombinación (4, 5 y 6) y se incrementa la longitud de cada individuo a 75 tonos relativos (es decir, cada individuo ) crece un 50% en tamaño (se ejecutan 30 procesos evolutivos de 1000 generaciones cada uno, con 500 individuos por población y 75 tonos relativos por individuo).
</div>

<div align="center">

![](./plots/plot_indsize75_popsize500_gens1100_multi_strategy_runs30_elitism_random.png)  
<b>Figura 9.</b> Mismo experimento que el representado en la Figura 8, pero incrementando el número total de generaciones a 1100 en lugar de 1000 (se ejecutan 30 procesos evolutivos de 1100 generaciones cada uno, con 500 individuos por población y 75 tonos relativos por individuo).
</div>

## Análisis de resultados


## Conclusión


## Bibliografía

[1] [Understanding Basic Music Theory](https://www.opentextbooks.org.hk/system/files/export/2/2180/pdf/Understanding_Basic_Music_Theory_2180.pdf)

[2] [A simple genetic algorithm for music generation by means of algorithmic information theory](https://www.researchgate.net/profile/Manuel-Alfonseca/publication/221008730_A_simple_genetic_algorithm_for_music_generation_by_means_of_algorithmic_information_theory/links/02e7e521b9152b11e4000000/A-simple-genetic-algorithm-for-music-generation-by-means-of-algorithmic-information-theory.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImluZGV4IiwicGFnZSI6InB1YmxpY2F0aW9uIn19)

[3] [Music Recombination using a Genetic Algorithm](https://scholarworks.indianapolis.iu.edu/bitstream/handle/1805/21281/Majumder_2019_music.pdf?isAllowed=y&sequence=1)

[4] [Using general-purpose compression algorithms for music analysis](https://vbn.aau.dk/files/223712009/rapport.pdf)

[5] [LZ77 Is All You Need? Why Gzip + KNN Works for Text Classification](https://blog.codingconfessions.com/p/lz77-is-all-you-need)

[6] [Compression: Clearing the Confusion on ZIP, GZIP, Zlib and DEFLATE](https://dev.to/biellls/compression-clearing-the-confusion-on-zip-gzip-zlib-and-deflate-15g1)

[7] [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426.pdf)

