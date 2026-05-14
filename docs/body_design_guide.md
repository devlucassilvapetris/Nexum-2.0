# Nexum v2.0 - Guia de Design do Corpo Robótico

## Visão Geral

Este guia explica como projetar e moldar o corpo robótico do Nexum v2.0, abordando desde a seleção de materiais até a fabricação final.

## 1. Arquitetura Estrutural

### 1.1 Segmentos do Corpo

O corpo do Nexum v2.0 é dividido em segmentos modulares:

- **Cabeça**: Processamento de áudio, visão e sensores
- **Pescoço**: Conexão flexível entre cabeça e tronco
- **Tronco**: Unidade central com CPU, GPU e baterias
- **Cintura**: Conexão entre tronco e pernas
- **Braços**: Manipulação e interação
- **Pernas**: Locomoção e suporte

### 1.2 Sistema de Materiais

```python
# Materiais disponíveis e suas propriedades
materiais = {
    "fibra_de_carbono": {
        "densidade": 1600,  # kg/m³
        "resistencia": 3500,  # MPa
        "custo": 3.0,  # Fator de custo
        "imprimivel": False
    },
    "aluminio": {
        "densidade": 2700,
        "resistencia": 276,
        "custo": 1.0,
        "imprimivel": False
    },
    "abs_plastico": {
        "densidade": 1050,
        "resistencia": 40,
        "custo": 0.3,
        "imprimivel": True
    }
}
```

## 2. Processo de Design

### 2.1 Design Inicial

```python
from nexum.hardware.structure import BodyDesigner

# Criar designer
designer = BodyDesigner()

# Projetar corpo humanoide (1.75m, 75kg)
design = designer.design_humanoid_body(
    height=1.75,
    weight_target=75
)

# Exportar design
designer.export_design("nexum_body_design.json")
```

### 2.2 Considerações de Design

#### Peso e Equilíbrio
- Centro de massa deve estar entre 10cm do design ideal
- Distribuição de peso: 40% tronco, 30% pernas, 20% braços, 10% cabeça
- Peso total alvo: 70-80kg

#### Resistência Estrutural
- Fator de segurança mínimo: 3.0
- Resistência à fadiga para movimentos repetitivos
- Proteção contra impactos

#### Integração de Componentes
- Espaço para CPU, GPU, baterias
- Sistema de refrigeração
- Passagem de cabos e conectores

## 3. Fabricação

### 3.1 Impressão 3D

Componentes imprimíveis:
- Segmentos externos (carcaça)
- Suportes de componentes
- Protótipos e testes

```python
# Configuração para impressão 3D
print_settings = {
    "material": "PETG",
    "infill": "40%",
    "layer_height": "0.2mm",
    "supports": True,
    "resolution": "0.1mm"
}
```

### 3.2 Usinagem CNC

Componentes metálicos:
- Estrutura principal
- Juntas e conexões críticas
- Suportes de motores

### 3.3 Montagem

#### Sequência de Montagem
1. **Núcleo do Tronco**: Instalar componentes principais
2. **Unidade da Cabeça**: Montar sensores e processamento
3. **Braços**: Conectar ao tronco
4. **Pernas**: Montar sistema de locomoção
5. **Integração Final**: Conectar todos os sistemas

#### Tolerâncias Críticas
- Furos de montagem: ±0.1mm
- Alinhamento de componentes: ±0.5mm
- Distribuição de peso: manter centro de massa

## 4. Materiais Recomendados

### 4.1 Por Segmento

| Segmento | Material Primário | Justificativa |
|----------|-------------------|---------------|
| Cabeça | PETG/ABS | Leve, fácil de imprimir, bom para sensores |
| Pescoço | Alumínio | Resistência com peso moderado |
| Tronco | Fibra de Carbono | Máxima resistência com peso mínimo |
| Cintura | Alumínio | Boa resistência, fácil usinagem |
| Braços | PETG | Flexibilidade, peso reduzido |
| Pernas | Alumínio/Carbono | Resistência estrutural crítica |

### 4.2 Cálculo de Materiais

```python
# Exemplo de cálculo de material para o tronco
tronco_dimensions = (0.4, 0.6, 0.2)  # largura, altura, profundidade
wall_thickness = 0.006  # 6mm
material_density = 1600  # fibra de carbono

# Volume do material
volume_exterior = tronco_dimensions[0] * tronco_dimensions[1] * tronco_dimensions[2]
volume_interior = (tronco_dimensions[0] - 2*wall_thickness) * \
                  (tronco_dimensions[1] - 2*wall_thickness) * \
                  (tronco_dimensions[2] - 2*wall_thickness)
volume_material = volume_exterior - volume_interior

# Peso
peso_tronco = volume_material * material_density
print(f"Peso do tronco: {peso_tronco:.2f} kg")
```

## 5. Integração com Sistemas

### 5.1 Integração Mecânica

```python
from nexum.hardware.kinematics import MotorController
from nexum.hardware.structure import MechanicalController

# Controladores
motor_controller = MotorController()
mechanical_controller = MechanicalController()

# Configurar motores para cada junta
motor_configs = {
    "ombro_direito": MotorConfig(
        id="shoulder_right",
        name="Right Shoulder Motor",
        type=MotorType.SERVO,
        max_speed=10.0,
        max_torque=15.0
    ),
    "cotovelo_direito": MotorConfig(
        id="elbow_right", 
        name="Right Elbow Motor",
        type=MotorType.SERVO,
        max_speed=8.0,
        max_torque=10.0
    )
}

# Adicionar motores
for config in motor_configs.values():
    motor_controller.add_motor(config)
```

### 5.2 Integração de Sensores

- **Sensores de pressão**: Nos pés para detecção de contato
- **IMU**: No tronco para detecção de orientação
- **Encoders**: Nas juntas para controle preciso
- **Sensores de temperatura**: Monitoramento térmico

## 6. Teste e Validação

### 6.1 Testes Estruturais

```python
# Teste de carga
def test_structural_load():
    # Simular carga máxima
    max_load = 150  # kg
    
    # Verificar deformação
    deformation = calculate_deformation(max_load)
    assert deformation < 0.01  # Máximo 1cm de deformação
    
    # Verificar tensão
    stress = calculate_stress(max_load)
    yield_strength = get_material_yield_strength()
    assert stress < yield_strength / 3  # Fator de segurança 3

# Teste de fadiga
def test_fatigue():
    # Simular 1 milhão de ciclos
    for i in range(1000000):
        apply_load_cycle()
        if i % 100000 == 0:
            check_for_cracks()
```

### 6.2 Testes de Movimento

- **Alcance de movimento**: Verificar limites das juntas
- **Velocidade máxima**: Testar velocidades operacionais
- **Precisão**: Verificar posicionamento preciso
- **Estabilidade**: Testar equilíbrio dinâmico

## 7. Manutenção e Upgrades

### 7.1 Manutenção Preventiva

- Inspeção visual semanal
- Lubrificação de juntas mensal
- Teste de sensores quinzenal
- Calibração de motores trimestral

### 7.2 Sistema Modular

Design modular permite:
- Substituição fácil de componentes
- Upgrades incrementais
- Reparos localizados
- Expansão de capacidades

## 8. Considerações de Segurança

### 8.1 Segurança Estrutural

- Limites de torque em todas as juntas
- Paradas de emergência
- Detecção de colisões
- Monitoramento de integridade estrutural

### 8.2 Segurança Operacional

- Áreas de trabalho seguras
- Protocolos de emergência
- Treinamento de operadores
- Sistemas de backup

## 9. Exemplo Prático

### 9.1 Design Completo

```python
# Design completo do corpo
from nexum.hardware.structure import BodyDesigner

designer = BodyDesigner()

# Design personalizado
design = designer.design_humanoid_body(
    height=1.80,  # 180cm
    weight_target=78  # 78kg
)

# Gerar arquivos de fabricação
designer.generate_manufacturing_files("fabricacao")

# Resultados
print(f"Altura total: {design['total_height']:.2f}m")
print(f"Peso total: {design['total_weight']:.2f}kg")
print(f"Centro de massa: {design['center_of_mass']}")
```

### 9.2 Simulação de Movimento

```python
# Simular movimento de caminhada
def simulate_walking():
    # Sequência de movimentos
    movements = [
        ("left_hip", 30),   # Levantar quadril esquerdo
        ("left_knee", 45),  # Dobrar joelho esquerdo
        ("left_ankle", 15), # Mover tornozelo esquerdo
        ("right_hip", -30), # Mover quadril direito
        # ... sequência completa
    ]
    
    for joint, angle in movements:
        mechanical_controller.set_joint_position(joint, math.radians(angle))
        time.sleep(0.1)  # 100ms por movimento
```

## 10. Próximos Passos

1. **Protótipo Inicial**: Criar protótipo em escala reduzida
2. **Testes de Materiais**: Validar propriedades mecânicas
3. **Integração de Sistema**: Conectar hardware e software
4. **Otimização**: Refinar design baseado em testes
5. **Produção**: Escalar para fabricação completa

---

Este guia fornece o framework completo para projetar e fabricar o corpo robótico do Nexum v2.0, garantindo integração perfeita com os sistemas existentes e preparando para futuras evoluções.
