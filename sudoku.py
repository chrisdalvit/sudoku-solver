import numpy as np

class Sudoku:
    
    def __init__(self, cells) -> None:
        if len(cells) != 81:
            raise ValueError(f"Expected a list of 81 cells. Got list of {len(cells)} cells.")
        self.cells = np.array(cells)
    
    def _is_row_complete(self, row):
        return set(self[row,:]) == {1,2,3,4,5,6,7,8,9}
        
    def _is_column_complete(self, col):
        return set(self[:,col]) == {1,2,3,4,5,6,7,8,9}
    
    def _is_block_complete(self, block):
        i, j = block // 3, block % 3    
        return set(self[3*i:3*i+3,3*j:3*j+3].reshape(-1)) == {1,2,3,4,5,6,7,8,9}
    
    def is_solved(self):
        return all(self._is_row_complete(idx) and 
                   self._is_column_complete(idx) and 
                   self._is_block_complete(idx) for idx in range(9))
        
    def _get_block(self, i, j):
        row, column = i // 3, j // 3
        return self[3*row:3*row+3, 3*column:3*column+3]
        
    def _compute_empty_squares(self):
        empty_squares = []
        for i in range(9):
            for j in range(9):
                if self[i,j] is None:
                    candidates = []
                    for k in range(1,10):
                        if k not in self[i,:] and k not in self[:,j] and k not in self._get_block(i,j):
                            candidates.append(k)
                    empty_squares.append(((i,j), candidates))
        return empty_squares
                
        
    def __getitem__(self, key):
        if type(key) == tuple:
            return self.cells.reshape((9,9))[key]
        return self.cells[key]
    
    def __setitem__(self, key, newvalue):
        if type(key) == tuple:
            self.cells.reshape((9,9))[key] = newvalue
        else:
            self.cells[key] = newvalue
    
    def __repr__(self) -> str:
        repr = []
        for i in range(9):
            row = [ "  |" if e is None else str(e) + " |" for e in self[i,:]]
            row.append("\n")
            repr.append(" ".join(row))
        return "".join(repr)
    
    def solve(self):
        if self.is_solved():
            return self
        else:
            empty_squares = self._compute_empty_squares()
            if len(empty_squares) == 0:
                return None
            sorted_empty_squares = sorted(empty_squares, key=lambda x: len(x[1]))
            (i,j), candidates  = sorted_empty_squares[0]
            for c in candidates:
                self[i,j] = c
                return self.solve()
            return None
                
            