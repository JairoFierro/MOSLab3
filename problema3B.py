# Definir las variables simbólicas 
x, y, z, w = sp.symbols('x y z w')


# Se define la función  f(x, y, z, w) = x^4 + y^4 + z^4 + w^4 - 4(x + y + z + w) simbólicamente
f_sym = x**4 + y**4 + z**4 + w**4 - 4*(x + y + z + w)   

