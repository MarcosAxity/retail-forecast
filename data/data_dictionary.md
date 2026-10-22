# Diccionario de Datos

---

## `transactions.csv`

Transacciones diarias agregadas por tienda y categoría de producto.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | string (YYYY-MM-DD) | Fecha de la transacción |
| `store_id` | string | Identificador único de la tienda (ej. `STR_001`) |
| `category` | string | Categoría de producto |
| `total_transactions` | int | Número total de transacciones del día |
| `cash_transactions` | int | Transacciones pagadas en efectivo. Puede contener nulos. |
| `card_transactions` | int | Transacciones pagadas con tarjeta |
| `amount_total` | float | Monto total vendido en MXN |
| `amount_cash` | float | Monto total cobrado en efectivo (MXN). Puede contener nulos. |
| `amount_card` | float | Monto total cobrado con tarjeta (MXN) |
| `units_sold` | float | Unidades vendidas. Puede contener nulos. |
| `avg_ticket` | float | Ticket promedio por transacción (MXN). Puede contener nulos. |
| `has_promotion` | int (0/1) | Indica si hubo promoción activa en esa tienda/categoría/día |
| `replenishment_signal` | float | Señal generada por el sistema de reposición de inventario de la tienda, calculada internamente con base en la demanda observada. Puede contener nulos en los últimos días del periodo. |

**Categorías disponibles:** `Abarrotes`, `Bebidas`, `Cuidado_Personal`, `Hogar`, `Electronica`, `Ropa`

**Nota de calidad de datos:** El dataset contiene datos faltantes en algunas columnas como resultado de fallas en sistemas de punto de venta y problemas de conectividad. Algunas tiendas tienen periodos completos sin datos.

---

## `stores.csv`

Características estáticas de cada tienda.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `store_id` | string | Identificador único de la tienda |
| `store_format` | string | Formato de la tienda: `Supercenter`, `Bodega`, `Express` |
| `region` | string | Región geográfica: `Norte`, `Centro`, `Sur`, `Occidente`, `Oriente` |
| `size_sqm` | int | Tamaño de la tienda en metros cuadrados |
| `num_checkouts` | int | Número de cajas registradoras |
| `opening_year` | int | Año de apertura de la tienda |
| `socioeconomic_level` | string | Nivel socioeconómico predominante del área: `C`, `C+`, `B`, `A/B` |
| `has_pharmacy` | bool | Si la tienda tiene farmacia |
| `has_fuel_station` | bool | Si la tienda tiene gasolinera |

---

## `calendar.csv`

Variables temporales y eventos para cada día del periodo.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | string (YYYY-MM-DD) | Fecha |
| `day_of_week` | int | Día de la semana (0=Lunes, 6=Domingo) |
| `day_name` | string | Nombre del día en inglés |
| `week_of_year` | int | Semana del año (ISO) |
| `month` | int | Mes (1–12) |
| `year` | int | Año |
| `quarter` | int | Trimestre (1–4) |
| `season` | string | Estación: `Invierno`, `Primavera`, `Verano`, `Otoño` |
| `is_holiday` | bool | Si es día festivo oficial en México |
| `holiday_name` | string / null | Nombre del festivo, si aplica |
| `is_payday` | bool | Si corresponde a una quincena (día 15 o último día del mes) |
| `is_weekend` | bool | Si es sábado o domingo |
| `is_navidad_season` | bool | Si está dentro del periodo navideño (15 dic – 6 ene) |
| `is_buen_fin` | bool | Si corresponde a los días del Buen Fin |
| `is_semana_santa` | bool | Si corresponde a Semana Santa |

---

## Periodo de cobertura

`2023-01-01` a `2024-02-29` (~14 meses)
