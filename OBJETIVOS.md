# Objetivos del speedrunner de Super Mario Bros. Deluxe

Usa este texto como contexto al trabajar en este proyecto: define qué buscamos y qué no.

## Misión
Entrenar con aprendizaje por refuerzo (PPO + PyBoy) una IA que pase cada nivel de **Super Mario Bros.
Deluxe (Game Boy Color, modo Original 1985)** en el **menor número de fotogramas posible**, y que descubra por sí misma rutas, movimientos
o trucos que un jugador humano no usaría. El objetivo final es el juego completo (1-1 a 8-4) con el
mejor tiempo total.

## Qué cuenta como éxito, en orden
1. **Llegar a la meta** del nivel de forma fiable (≥ 80 % de los episodios).
2. **Bajar el tiempo**: cada récord de `records.json` es una partida real y repetible
   (`ver_ia.py --record`). Medimos en fotogramas, no en el reloj del juego.
3. **Descubrir algo inesperado**: un atajo, un salto que se salta un tramo, un uso raro de enemigos,
   tuberías o la física. Si aparece, se documenta con el récord que lo contiene.
4. Repetirlo en cada nivel y después encadenar el juego completo.

## Reglas de diseño (no negociables)
- **Decir qué, nunca por dónde.** La recompensa solo premia avanzar más que nunca, llegar a la meta y
  hacerlo antes. Nada de coordenadas, rutas ni guías escritas a mano: impedirían que encuentre atajos.
- **Control fino.** Acciones de pocos fotogramas (`FRAMES_POR_ACCION`) y combinaciones de botones: los
  trucos necesitan precisión.
- **Partidas repetibles.** Todo récord guarda sus acciones y empieza en un state conseguido jugando. Si un cambio rompe la
  repetición exacta, es un fallo.
- **Las trampas de la recompensa no valen.** Si la IA saca puntos sin acercarse a la meta (quedarse
  quieta sin morir, ir y volver, aprovechar una lectura rara de la RAM), es un fallo nuestro: se corrige
  la recompensa, no se celebra. Los trucos del juego sí valen: lo que importa es que la meta llegue antes.
- **Sin modificar la ROM ni la RAM** durante el entrenamiento (nada de vidas infinitas o saltar niveles
  escribiendo memoria). Solo se leen datos.

## Cómo se trabaja
- Antes de cambiar recompensas, explica el problema con datos (`analisis.py`, `metrics.jsonl`,
  TensorBoard) y comprueba el cambio en el emulador.
- Las pruebas no escriben `records.json` (`MARIO_SIN_RECORDS=1`).
- No lanzar ni parar entrenamientos que haya lanzado el usuario sin preguntar; para parar, el archivo
  `PARAR`.
- Cada cambio con sus tests y en un commit propio.

## Métricas a vigilar
- % de episodios que llegan a la meta y cómo terminan los demás (muerte, atasco).
- Segundos hasta la meta: mejor, mediana y tendencia.
- Avance medio en píxeles mientras aún no llega a la meta.
- Estabilidad del aprendizaje: `approx_kl`, `explained_variance`, entropía.
