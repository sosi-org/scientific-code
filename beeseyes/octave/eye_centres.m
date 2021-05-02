% eyepoints.m
function retval = eye_centres(n)

  nx = floor(sqrt(n)+1)
  ny = floor(sqrt(n))

  xa = linspace(0,nx-1,nx);
  ya = linspace(0,ny-1,ny);
  [X,Y] = meshgrid(xa,ya);
  X=X(:);
  Y=Y(:);
  c_xyi = [X,Y];
  c_xy  = c_xyi + (rand(size(c_xyi))-0.5) * 0.8;
  c_xy
  retval = c_xy;
  % eye_centres = rand(2,n);
endfunction
