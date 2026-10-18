# Laboratorio 09 – Visualización Estática

CC3084 Data Science · Universidad del Valle de Guatemala · Semestre II 2026
Integrantes: José López (23773) · Hugo Barillas (23306) · Luis Palacios (23933)

Infografía de una página que resume el Laboratorio 1 de Series de Tiempo
(Base de Migración 2009–2026, INGUAT/IGM) para un directivo del INGUAT.

## Cómo generarla

```bash
pip install -r requirements.txt
python infografia.py
```

Salidas en `infografia/`:
- `infografia.png` — 3300 × 5100 px (11 × 17 pulgadas a 300 dpi)
- `infografia.pdf` — una página de 11 × 17 pulgadas

## Estructura

- `infografia.py` — script completo; cada bloque de la infografía es una función
  (`bloque_titular`, `bloque_panorama`, `bloque_estacionalidad`, `bloque_modelos`,
  `bloque_recomendaciones`, `pie`).
- `docs/` — documento de explicación de las decisiones de diseño (PDF).
- `data/` — colocar aquí `Base_Migracion_2009-2026jun.xlsx` (no se versiona; uso académico).
- `fonts/` — Source Sans 3 (licencia SIL Open Font License).

El pronóstico reproduce el Prophet del Laboratorio 1 (estacionalidad multiplicativa,
corte 70/30, intervalo del 80%): MAE 136,571 y RMSE 144,939 sobre la serie total.

Los datos se comparten con fines estrictamente académicos; este repositorio debe ser privado.
