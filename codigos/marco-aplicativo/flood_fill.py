@njit
def ff(mat, i, j, x1, y1, x2, y2, directions):
  if (0 <= i < mat.shape[0]) and (0 <= j < mat.shape[1]) and mat[i, j] != 0:
    mat[i, j] = 0
    x1 = min(x1, j)
    y1 = min(y1, i)
    x2 = max(x2, j)
    y2 = max(y2, i)

    for dx, dy in directions:
      x1, y1, x2, y2 = ff(mat, i+dy, j+dx, x1, y1, x2, y2, directions)
    
  return x1, y1, x2, y2
