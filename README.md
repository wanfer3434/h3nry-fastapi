<div class="button" style="background-color: #007bff;">Data Science</div>
<div class="button" style="background-color: #28a745;">Machine Learning</div>
<div class="button" style="background-color: #dc3545;">Natural Language Processing</div>
<div class="button" style="background-color: #ffc107;">FastAPI</div>
<div class="button" style="background-color: #17a2b8;">Docker</div>
<div class="button" style="background-color: #6610f2;">Render</div>
<div class="button" style="background-color: #007bff;">Python</div>
<div class="button" style="background-color: #28a745;">pandas</div>
<div class="button" style="background-color: #dc3545;">NumPy</div>
<div class="button" style="background-color: #ffc107;">scikit-learn</div>
<div class="button" style="background-color: #17a2b8;">NLTK</div>
<div class="button" style="background-color: #6610f2;">spaCy</div>
<div class="button" style="background-color: #007bff;">Matplotlib</div>
<div class="button" style="background-color: #28a745;">Parquet</div>
<style>
.button {
    display: inline-block;
    padding: 10px 20px;
    margin: 5px;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
</style>

## Tabla de Contenidos

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Propósito](#propósito)
3. [Aspectos Destacados](#aspectos-destacados)
4. [Ecosistema Tecnológico](#ecosistema-tecnológico)
5. [Secuencia Operativa](#secuencia-operativa)
6. [Impacto](#impacto)
7. [Fuentes de Datos](#fuentes-de-datos)
8. [Extracción, Transformación y Carga (ETL)](#extracción-transformación-y-carga-etl)
9. [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
10. [Creación de Funciones](#creación-de-funciones)
11. [Modelado de Aprendizaje Automático](#modelado-de-aprendizaje-automático)
    - [Recomendación de Juegos](#recomendación-de-juegos)
    - [Recomendación de Usuario](#recomendación-de-usuario)
12. [Despliegue](#despliegue)
13. [Video](#video)
14. [Conclusiones](#conclusiones)
15. [Enlaces](#enlaces)


## Resumen
En un mundo saturado de opciones de entretenimiento digital, el Sistema de Recomendación de Juegos de Steam resalta como un proyecto pionero en la implementación de ciencia de datos y aprendizaje automático para mejorar la experiencia de usuario en la selección de videojuegos digitales. Más que una exhibición de habilidades técnicas, este proyecto refleja una comprensión profunda del comportamiento y las preferencias de los jugadores.

## Propósito
Los objetivos del proyecto se centran en dos áreas principales: análisis de sentimientos y recomendación de juegos. Mediante técnicas de Procesamiento de Lenguaje Natural (NLP), se analizan los comentarios de los usuarios para obtener insights valiosos, mientras que el sistema de recomendación utiliza algoritmos de aprendizaje automático para ofrecer sugerencias personalizadas, aumentando así la satisfacción y participación del usuario.

## Ecosistema Tecnológico
El proyecto se desarrolla con herramientas tecnologicas, que incluye Python, TensorFlow, FastAPI, Docker, entre otros. 

## Secuencia Operativa

1. **Agregación de Datos**
   - Recopilación de datos de juegos de varias fuentes dentro de la plataforma Steam.

2. **Limpieza y Preprocesamiento de Datos**
   - Limpieza y transformación de datos sin procesar para garantizar que sean adecuados para el análisis.

3. **Análisis Exploratorio de Datos (EDA)**
   - Obtener información de los datos a través de la visualización y el análisis estadístico.
   - Identificación de patrones y tendencias en los conjuntos de datos.

4. **Análisis de Sentimientos de Revisión de Usuarios**
   - Aplicación de técnicas de PNL para comprender los sentimientos de los usuarios a partir de comentarios y reseñas.

5. **Modelo de Aprendizaje Automático**
   - Implementación de un modelo de Machine Learning, incluyendo el uso de Cosine Similarity, para recomendar juegos basados en las preferencias del usuario.
   - Desarrollo de funciones para la recomendación de juegos basada en la similitud de coseno y el historial de interacciones del usuario.

6. **Desarrollo API**
   - Creación de una API para proporcionar recomendaciones de juegos en tiempo real a los usuarios.
   - Integración de las funciones de recomendación de juegos dentro de la API para acceso y uso por parte de los usuarios.

7. **Insights Centricos en el Usuario**
   - Análisis del comportamiento del usuario y métricas de participación del usuario.
   - Procesamiento de datos centrado en el usuario para mejorar la experiencia del usuario y las recomendaciones de juegos.

8. **Toma de Decisiones Impulsada por Datos**
   - Empoderamiento de la toma de decisiones a través de ideas basadas en datos y experiencias de juego personalizadas.

9. **Extracción, Transformación y Carga (ETL)**
   - Extracción de datos de archivos JSON iniciales.
   - Transformación de datos mediante limpieza y estructuración.
   - Carga de datos limpios y transformados en formato 'parquet' para optimización de almacenamiento y recuperación.

10. **Despliegue**
    - Implementación en la plataforma Render a través de Docker Hub para hacer accesible el sistema de recomendación de juegos.

## Video
[Ver demostración en video](https://youtu.be/J9CmQtHPLII)


## Conclusión
Estos puntos aborda cada etapa crucial del proyecto, desde la recopilación y preparación de datos hasta la implementación y despliegue del sistema de recomendación de juegos. Cada paso contribuye al objetivo general de mejorar la experiencia de juego para los usuarios de la plataforma Steam.

Finalmente, el proyecto se despliega en la plataforma Render a través de Docker Hub, garantizando así su accesibilidad para los usuarios. Esta implementación exitosa del Producto Mínimo Viable (MVP, por sus siglas en inglés) representa un logro significativo, aunque se reconoce la necesidad de continuar optimizando y mejorando la eficiencia del sistema en el futuro.





