import uaibot as ub
import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def compute_control_quaternion(q_prev, q, q_next):
    """
    Calcula o quaternion de controle para q, usando os quaternions adjacentes.
    
    q_prev, q, q_next são objetos Rotation.
    Utiliza a fórmula:
        control = q * exp( - ( log(q⁻¹*q_next) + log(q⁻¹*q_prev) )/4 )
    """
    log1 = (q.inv() * q_next).as_rotvec()
    log2 = (q.inv() * q_prev).as_rotvec()
    avg = -0.25 * (log1 + log2)
    control = q * Rotation.from_rotvec(avg)
    return control

def squad(q1, q2, a, b, t):
    """
    Interpolação SQUAD entre q1 e q2, usando os quaternions de controle a e b.
    
    q1, q2, a e b são objetos Rotation e t é o parâmetro (0 <= t <= 1).
    A fórmula é:
       SQUAD(q1,q2,a,b;t) = SLERP( SLERP(q1,q2;t), SLERP(a,b;t), 2*t*(1-t) )
    """
    key_times = [0, 1]
    rot_main = Rotation.from_quat([q1.as_quat(), q2.as_quat()])
    slerp_main = Slerp(key_times, rot_main)([t])[0]
    
    rot_control = Rotation.from_quat([a.as_quat(), b.as_quat()])
    slerp_control = Slerp(key_times, rot_control)([t])[0]
    
    final_param = 2 * t * (1 - t)
    rot_final = Rotation.from_quat([slerp_main.as_quat(), slerp_control.as_quat()])
    final_rot = Slerp(key_times, rot_final)([final_param])[0]
    
    return final_rot

def composite_squad(R_prev, R0, R1, R2, R_next, t):
    """
    Recebe 5 matrizes de rotação:
      - R_prev: matriz de rotação do quaternion vizinho anterior (extrapolado ou fornecido)
      - R0: rotação inicial
      - R1: rotação intermediária
      - R2: rotação final
      - R_next: matriz de rotação do quaternion vizinho após o final
    e um parâmetro global t em [0,1].
    
    Para t global:
      - se t <= 0.5, interpolamos entre R0 e R1 (segmento 1);
      - se t > 0.5, interpolamos entre R1 e R2 (segmento 2).
    
    Retorna a matriz de rotação interpolada.
    """
    # Converte as matrizes em objetos Rotation
    q_prev = Rotation.from_matrix(R_prev)
    q0 = Rotation.from_matrix(R0)
    q1 = Rotation.from_matrix(R1)
    q2 = Rotation.from_matrix(R2)
    q_next = Rotation.from_matrix(R_next)
    
    if t <= 0.5:
        # Segmento entre q0 (inicial) e q1 (intermediária)
        u = 2 * t  # mapeia t de [0,0.5] para [0,1]
        a0 = compute_control_quaternion(q_prev, q0, q1)
        b0 = compute_control_quaternion(q0, q1, q2)
        q_interp = squad(q0, q1, a0, b0, u)
    else:
        # Segmento entre q1 (intermediária) e q2 (final)
        u = 2 * (t - 0.5)  # mapeia t de [0.5,1] para [0,1]
        a1 = compute_control_quaternion(q0, q1, q2)
        b1 = compute_control_quaternion(q1, q2, q_next)
        q_interp = squad(q1, q2, a1, b1, u)
    
    return q_interp.as_matrix()

def get_configuration(robot):
  return robot.q

def set_configuration_speed(robot, qdot_des):
  q_next = robot.q + qdot_des*dt
  robot.add_ani_frame(time = t+dt, q = q_next)

def fun_F(r, k):
  return -k*r

def quadratic_spline_3D(s0, s1, s2, t):
    
    b = 4 * (s1 - s0) - (s2 - s0)
    a = (s2 - s0) - b
    c = s0
    
    t = np.array(t).reshape(1, -1)
    s_t = a @ (t**2) + b @ t + c
    
    return s_t


###############################################

robot = ub.Robot.create_kuka_kr5(ub.Utils.trn([0,0,0.2]))

texture_table = ub.Texture(
    url='https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/Textures/rough_metal.jpg',
    wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[1, 1])

material_table = ub.MeshMaterial(texture_map=texture_table, roughness=1, metalness=1, opacity=1)
table1 = ub.Box(name="table1", htm = ub.Utils.trn([0.8,0,0.15]), width=0.5, depth=0.5, height=0.4, mesh_material=material_table)
table2 = ub.Box(name="table2", htm = ub.Utils.trn([0,-0.8,0.15]), width=0.5, depth=0.5, height=0.4, mesh_material=material_table)
table3 = ub.Box(name="table3", htm = ub.Utils.trn([0,0,0.1]), width=0.3, depth=0.3, height=0.2, mesh_material=material_table)
obstacle = ub.Box(name="obstacle", htm = ub.Utils.trn([0.8,-0.8,0.5]), width=0.6, depth=0.6, height=1, mesh_material=material_table)

material_cube = ub.MeshMaterial(roughness=1, metalness=1, opacity=1, color="purple")
cube = ub.Box(name="cube", htm = ub.Utils.trn([0.8,0,0.4]), width=0.1, depth=0.1, height=0.1, mesh_material=material_cube)

###############################################

initial_H = robot.fkm()

dt = 0.01
t = 0
tmax = 25

cube_htm = ub.Utils.trn([0.8,0,0.45]) * ub.Utils.roty(np.pi)
ball_tr = ub.Ball(htm = np.identity(4), radius=0.02, color="cyan")

s_cube = cube_htm[0:3 , 3]
x_cube = cube_htm[0:3 , 0]
y_cube = cube_htm[0:3 , 1]
z_cube = cube_htm[0:3 , 2]

J_r = np.matrix(np.zeros((6,6)))
r   = np.matrix(np.zeros((6,1)))

frame_get_cube = ub.Frame(htm = cube_htm)


for i in range(round(0/dt),round(tmax/dt)):
  
  #################################
  # Início da lógica de controle  #
  #################################

  jg , fkm = robot.jac_geo()
  q = get_configuration(robot)

  x_eef = fkm[0:3 , 0]
  y_eef = fkm[0:3 , 1]
  z_eef = fkm[0:3 , 2]
  s_eef = fkm[0:3 , 3]

  if (np.linalg.norm(s_eef - s_cube) < 0.01 and np.linalg.norm(z_eef - z_cube) < 0.06):
    robot.attach_object(cube)
    break

  J_v = jg[0 : 3 , : ]
  J_w = jg[3 : 6 , : ]

  r[0 : 3] = s_eef - s_cube
  r[3] = 0 # So importa a orientação do eixo Z
  r[4] = 0
  r[5] = 1 - z_cube.T*z_eef

  J_r[0 : 3, : ] = J_v
  J_r[3 , : ]  = np.zeros((1,6))
  J_r[4 , : ]  = np.zeros((1,6))
  J_r[5 , : ]  = z_cube.T*ub.Utils.S(z_eef)*J_w

  u = np.matrix(np.zeros((6,1)))
  u = ub.Utils.dp_inv_solve(A = J_r, b = (fun_F(r,2)), eps = 1e-3 )

  set_configuration_speed(robot , u)

  #################################
  # Fim da lógica de controle     #
  #################################
  t+=dt



#######################################
#   PONTOS USADOS DURANTE O TRAJETO   #
#######################################
H0 = robot.fkm()
Q0 = H0[ 0 : 3, 0 : 3]

H1 = ub.Utils.trn([0, -0.3, 0.85])*ub.Utils.roty(np.pi/2)* ub.Utils.rotx(np.pi/2)*ub.Utils.rotz(np.pi/2)
frame_h1 = ub.Frame(htm = H1)
Q1 = H1[ 0 : 3, 0 : 3]

H2 = ub.Utils.trn([0, -0.8, 0.45])* ub.Utils.rotx(np.pi)
frame_h2 = ub.Frame(htm = H2)
Q2 = H2[ 0 : 3, 0 : 3]

Q_prev = Q0
Q_next = Q2
#######################################

s0 = np.array([0.8, 0.0, 0.45]).reshape((3,1)) 
s1 = H1[ 0 : 3, 3]
s2 = H2[ 0 : 3, 3]
t_bg = t


Q_t = np.matrix(np.zeros((3,3)))
J_r = np.matrix(np.zeros((6, 6)))
r = np.matrix(np.zeros((6, 1)))

hist_r = np.matrix(np.zeros((6, 0)))
hist_u = np.matrix(np.zeros((6, 0)))
hist_t = []

for i in range( round(t_bg/dt) , round( (t + 15)/dt )):

  b = (t - t_bg)/12
  if b >= 1:
     b = 1

  #################################
  # Início da lógica de controle  #
  #################################

  q = get_configuration(robot)
  jg , fkm = robot.jac_geo()

  J_v = jg[0 : 3 , : ]
  J_w = jg[3 : 6 , : ]

  x_eef = fkm[0:3 , 0]
  y_eef = fkm[0:3 , 1]
  z_eef = fkm[0:3 , 2]
  s_eef = fkm[0:3 , 3]


  sd     = quadratic_spline_3D(s0, s1, s2, b)
  r_ant = r


  xd_prev = Q_t[0:3 , 0].reshape((3,1))
  yd_prev = Q_t[0:3 , 1].reshape((3,1))
  zd_prev = Q_t[0:3 , 2].reshape((3,1))
  Q_t    = composite_squad(Q_prev, Q0, Q1, Q2, Q_next, b)

  xd = Q_t[0:3 , 0].reshape((3,1))
  yd = Q_t[0:3 , 1].reshape((3,1))
  zd = Q_t[0:3 , 2].reshape((3,1))

  r[0 : 3] = s_eef - sd
  r[3] = 1 - xd.T*x_eef
  r[4] = 1 - yd.T*y_eef
  r[5] = 1 - zd.T*z_eef

  ff     = (r - r_ant)/dt

  J_r[0 : 3, : ] = J_v
  J_r[3 , : ]    = xd.T*ub.Utils.S(x_eef)*J_w
  J_r[4 , : ]    = yd.T*ub.Utils.S(y_eef)*J_w
  J_r[5 , : ]    = zd.T*ub.Utils.S(z_eef)*J_w
  u = ub.Utils.dp_inv_solve(A = J_r, b = (fun_F(r,2)), eps = 1e-3 )
  set_configuration_speed(robot, u)

  #################################
  # Fim da lógica de controle     #
  #################################

  hist_r = np.block([hist_r, r])
  hist_u = np.block([hist_u, u])
  hist_t.append(t)

  ball_tr.add_ani_frame(time = t, htm = ub.Utils.trn(quadratic_spline_3D(s0, s1, s2, b)))

  t += dt

robot.detach_object(cube)
                       

sim = ub.Simulation.create_sim_factory([robot, table1, table2, table3, obstacle, cube, frame_get_cube, frame_h1, frame_h2,ball_tr])
sim.run()
sim.save("/home/pedro55562/uaibot_files/code/", "controle02")