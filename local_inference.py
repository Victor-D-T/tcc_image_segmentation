"""
YOLO Duct Segmentation - Sistema de Identificação de Sessões de Dutos
Versão corrigida e otimizada para detecção de sessões em dutos quadrados de 50cm
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os
from pathlib import Path
from collections import defaultdict, deque
import json
import math

class DuctDetectorLocal:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Inicializa o detector de sessões de dutos
        
        Args:
            model_path: Caminho para o modelo YOLO treinado
            confidence_threshold: Limiar de confiança para detecções
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Carrega o modelo YOLO
        print(f"🔄 Carregando modelo de: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✅ Modelo carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
        
        # Configurações da câmera (valores padrão - ajustar conforme necessário)
        self.camera_matrix = None
        self.frame_width = None
        self.frame_height = None
        
        # Histórico de rastreamento
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.session_positions = {}  # Posições 3D das sessões
        self.camera_poses = [np.eye(4)]  # Histórico de poses da câmera
        
        # Métricas de performance
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        # CONSTANTES DO DUTO
        self.DUCT_SIZE = 0.50  # Duto quadrado de 50cm
        self.DUCT_HALF_SIZE = self.DUCT_SIZE / 2  # 25cm do centro
        self.DUCT_TOLERANCE = 0.02  # 2cm de tolerância
        
        # Constantes de movimento do drone
        self.MAX_DRONE_SPEED = 1.0  # m/s velocidade máxima
        self.MAX_FRAME_MOVEMENT = 0.05  # 5cm movimento máximo por frame
        
        # Configurações de sessões
        self.MIN_SESSION_SPACING = 0.8  # Espaçamento mínimo entre sessões (80cm)
        self.MAX_SESSION_SPACING = 1.5  # Espaçamento máximo entre sessões (150cm)
        self.SESSION_WIDTH_RATIO = 0.85  # Sessões ocupam ~85% da largura do duto
        self.MAX_SESSIONS_VISIBLE = 8  # Máximo de sessões visíveis simultaneamente
        
        # Direção atual do duto
        self.current_direction = np.array([0, 0, 1])  # Inicialmente para frente (Z)
        self.direction_history = deque(maxlen=10)
        
        print(f"🔧 Configurações do duto:")
        print(f"   Tamanho: {self.DUCT_SIZE}x{self.DUCT_SIZE}m")
        print(f"   Espaçamento de sessões: {self.MIN_SESSION_SPACING}-{self.MAX_SESSION_SPACING}m")
        print(f"   Máximo de sessões visíveis: {self.MAX_SESSIONS_VISIBLE}")
    
    def set_camera_calibration(self, frame_width, frame_height):
        """Define calibração da câmera baseada no tamanho do frame"""
        # Valores aproximados - idealmente calibrar com padrão de xadrez
        fx = fy = frame_width * 0.8  # Aproximação focal length
        cx = frame_width / 2
        cy = frame_height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"📷 Calibração da câmera definida:")
        print(f"   Resolução: {frame_width}x{frame_height}")
        print(f"   Focal length: {fx:.1f}")
    
    def estimate_depth_from_bbox(self, bbox_area, frame_width, frame_height):
        """
        Estima profundidade baseada no tamanho da bounding box
        
        Para sessões em duto de 50cm:
        - Muito próximo (30cm): sessão ocupa ~70% da largura
        - Próximo (60cm): sessão ocupa ~50% da largura
        - Médio (100cm): sessão ocupa ~30% da largura
        - Longe (150cm): sessão ocupa ~20% da largura
        """
        frame_area = frame_width * frame_height
        bbox_ratio = bbox_area / frame_area
        
        # Largura esperada da sessão (85% do duto = 42.5cm)
        session_width_m = self.DUCT_SIZE * self.SESSION_WIDTH_RATIO
        
        # Calcula profundidade baseada na proporção
        if bbox_ratio > 0.3:  # Muito próximo
            depth = 0.3
        elif bbox_ratio > 0.2:  # Próximo
            depth = 0.6
        elif bbox_ratio > 0.1:  # Médio
            depth = 1.0
        elif bbox_ratio > 0.05:  # Longe
            depth = 1.5
        else:  # Muito longe
            depth = 2.0
        
        # Limites físicos do duto
        depth = max(depth, 0.2)  # Mínimo 20cm
        depth = min(depth, 3.0)  # Máximo 3m (limite de visibilidade)
        
        return depth
    
    def pixel_to_world_constrained(self, center_2d, depth, camera_pose, frame_width, frame_height):
        """
        Converte coordenadas de pixel para mundo com restrições do duto
        """
        if self.camera_matrix is None:
            self.set_camera_calibration(frame_width, frame_height)
        
        # Parâmetros da câmera
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
        
        # Converte para coordenadas da câmera
        x_cam = (center_2d[0] - cx) * depth / fx
        y_cam = (center_2d[1] - cy) * depth / fy
        z_cam = depth
        
        # RESTRIÇÃO CRÍTICA: Limita às dimensões do duto
        x_cam = np.clip(x_cam, -self.DUCT_HALF_SIZE, self.DUCT_HALF_SIZE)
        y_cam = np.clip(y_cam, -self.DUCT_HALF_SIZE, self.DUCT_HALF_SIZE)
        
        # Converte para coordenadas mundiais
        cam_point = np.array([x_cam, y_cam, z_cam, 1])
        world_point = camera_pose @ cam_point
        
        return world_point[:3]
    
    def estimate_camera_motion(self, frame, prev_frame):
        """
        Estima movimento da câmera entre frames
        Movimento restrito pelas dimensões do duto
        """
        if prev_frame is None:
            return np.eye(4)
        
        # Movimento típico do drone no duto: principalmente para frente
        # Pequenos movimentos laterais e rotação permitidos
        
        # Movimento padrão para frente (ajustar baseado na velocidade do drone)
        forward_motion = 0.03  # 3cm por frame (ajustar conforme FPS do vídeo)
        
        transform = np.eye(4)
        transform[2, 3] = forward_motion  # Movimento no eixo Z (para frente)
        
        # Pequenas variações laterais devido ao voo do drone
        # (Pode ser refinado com análise de fluxo óptico)
        lateral_noise = np.random.normal(0, 0.005, 2)  # ±5mm de ruído
        transform[0, 3] = lateral_noise[0]  # X
        transform[1, 3] = lateral_noise[1]  # Y
        
        return transform
    
    def validate_session_positions(self, tracked_detections, camera_pose):
        """
        Valida se as posições das sessões fazem sentido no contexto do duto
        """
        valid_detections = []
        camera_pos = camera_pose[:3, 3]
        
        for detection in tracked_detections:
            track_id = detection['track_id']
            
            # Se já tem posição 3D, valida
            if track_id in self.session_positions:
                world_pos = self.session_positions[track_id]
                relative_pos = world_pos - camera_pos
                
                # Distância lateral do centro do duto
                lateral_distance = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
                
                # Deve estar dentro do duto (com tolerância)
                if lateral_distance <= (self.DUCT_HALF_SIZE + self.DUCT_TOLERANCE):
                    # Profundidade razoável
                    depth = relative_pos[2]
                    if 0.1 <= depth <= 3.0:
                        valid_detections.append(detection)
                        continue
            
            # Se não tem posição 3D ainda, aceita por enquanto
            valid_detections.append(detection)
        
        return valid_detections
    
    def process_frame(self, frame, frame_idx):
        """
        Processa um frame individual para detectar sessões
        """
        frame_height, frame_width = frame.shape[:2]
        
        if self.camera_matrix is None:
            self.set_camera_calibration(frame_width, frame_height)
            self.frame_width = frame_width
            self.frame_height = frame_height
        
        # Executa detecção/rastreamento YOLO
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            iou=0.3,  # IoU threshold
            max_det=self.MAX_SESSIONS_VISIBLE,
            verbose=False
        )
        
        # Anota o frame
        annotated_frame = results[0].plot()
        tracked_detections = []
        
        # Extrai informações das detecções
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                
                # Atualiza histórico
                self.track_history[track_id].append({
                    'frame': frame_idx,
                    'center': center,
                    'bbox': box,
                    'bbox_area': bbox_area,
                    'confidence': conf,
                    'class': int(cls)
                })
                
                tracked_detections.append({
                    'track_id': track_id,
                    'bbox': box,
                    'center': center,
                    'bbox_area': bbox_area,
                    'confidence': conf,
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
        
        return annotated_frame, tracked_detections
    
    def estimate_3d_positions(self, tracked_detections, camera_pose, frame_width, frame_height):
        """
        Estima posições 3D das sessões detectadas
        """
        for detection in tracked_detections:
            track_id = detection['track_id']
            center_2d = detection['center']
            bbox_area = detection['bbox_area']
            
            # Estima profundidade
            estimated_depth = self.estimate_depth_from_bbox(
                bbox_area, frame_width, frame_height)
            
            # Converte para coordenadas 3D
            world_pos = self.pixel_to_world_constrained(
                center_2d, estimated_depth, camera_pose, frame_width, frame_height)
            
            # Aplica suavização temporal
            if track_id in self.session_positions:
                prev_pos = self.session_positions[track_id]
                
                # Limita movimento entre frames
                movement = np.linalg.norm(world_pos - prev_pos)
                if movement > self.MAX_FRAME_MOVEMENT:
                    # Limita movimento máximo
                    direction = (world_pos - prev_pos) / movement
                    world_pos = prev_pos + direction * self.MAX_FRAME_MOVEMENT
                
                # Suavização (filtro passa-baixa)
                alpha = 0.1  # Fator de suavização
                self.session_positions[track_id] = (
                    alpha * world_pos + (1 - alpha) * prev_pos
                )
            else:
                self.session_positions[track_id] = world_pos
            
            print(f"📍 Sessão {track_id}: profundidade={estimated_depth:.2f}m, "
                  f"posição=({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")
    
    def draw_duct_info(self, frame, tracked_detections):
        """
        Desenha informações do duto e sessões no frame
        """
        h, w = frame.shape[:2]
        
        # Desenha limites do duto (aproximado)
        margin = 30  # pixels
        cv2.rectangle(frame, (margin, margin), (w-margin, h-margin), (100, 100, 100), 2)
        
        # Linha central do duto
        cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (100, 100, 100), 1)
        
        # Informações das sessões
        for detection in tracked_detections:
            track_id = detection['track_id']
            center = detection['center']
            confidence = detection['confidence']
            
            # Label da sessão
            if track_id in self.session_positions:
                world_pos = self.session_positions[track_id]
                depth = world_pos[2]
                label = f"S{track_id}: {depth:.2f}m ({confidence:.2f})"
            else:
                label = f"S{track_id}: nova ({confidence:.2f})"
            
            # Desenha label
            cv2.putText(frame, label, 
                       (int(center[0]), int(center[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Informações gerais
        info_text = [
            f"Duto: {self.DUCT_SIZE}x{self.DUCT_SIZE}m",
            f"Sessões detectadas: {len(self.session_positions)}",
            f"Confiança mínima: {self.confidence_threshold:.2f}",
            f"FPS: {self.avg_fps:.1f}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    def update_fps(self):
        """Atualiza cálculo de FPS"""
        self.fps_counter += 1
        if self.fps_counter % 10 == 0:  # Atualiza a cada 10 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.avg_fps = 10 / elapsed if elapsed > 0 else 0
            self.fps_start_time = current_time
    
    def save_reconstruction(self, output_path):
        """Salva dados de reconstrução 3D"""
        reconstruction_data = {
            'duct_size': self.DUCT_SIZE,
            'session_positions': {
                str(k): v.tolist() for k, v in self.session_positions.items()
            },
            'camera_poses': [pose.tolist() for pose in self.camera_poses],
            'total_sessions': len(self.session_positions)
        }
        
        with open(output_path, 'w') as f:
            json.dump(reconstruction_data, f, indent=2)
        
        print(f"💾 Dados de reconstrução salvos em: {output_path}")
    
    def run_video_inference(self, video_path, output_path=None, reconstruction_path=None, display=True):
        """
        Executa inferência em um vídeo com detecção de sessões de dutos
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Erro: Não foi possível abrir o vídeo: {video_path}")
            return
        
        # Propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processando vídeo:")
        print(f"   Arquivo: {video_path}")
        print(f"   Resolução: {frame_width}x{frame_height}")
        print(f"   FPS: {fps}")
        print(f"   Total de frames: {total_frames}")
        
        # Configuração de saída
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        prev_frame = None
        
        print("\n🚀 Iniciando processamento...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Estima movimento da câmera
            if prev_frame is not None:
                relative_pose = self.estimate_camera_motion(frame, prev_frame)
                current_pose = self.camera_poses[-1] @ relative_pose
                self.camera_poses.append(current_pose)
            
            # Processa frame
            annotated_frame, tracked_detections = self.process_frame(frame, frame_count)
            
            # Valida detecções
            valid_detections = self.validate_session_positions(
                tracked_detections, self.camera_poses[-1])
            
            # Estima posições 3D
            if valid_detections:
                self.estimate_3d_positions(
                    valid_detections, self.camera_poses[-1], frame_width, frame_height)
            
            # Desenha informações
            self.draw_duct_info(annotated_frame, valid_detections)
            
            # Salva frame
            if out:
                out.write(annotated_frame)
            
            # Exibe frame
            if display:
                cv2.imshow('Detecção de Sessões de Dutos', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Atualiza FPS
            self.update_fps()
            
            # Progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progresso: {progress:.1f}% ({frame_count}/{total_frames})")
            
            prev_frame = frame.copy()
        
        # Limpeza
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Salva reconstrução
        if reconstruction_path:
            self.save_reconstruction(reconstruction_path)
        
        # Relatório final
        print(f"\n✅ Processamento concluído!")
        print(f"📊 Estatísticas finais:")
        print(f"   Frames processados: {frame_count}")
        print(f"   Sessões detectadas: {len(self.session_positions)}")
        print(f"   FPS médio: {self.avg_fps:.1f}")
        
        return self.session_positions
    
    def run_webcam_inference(self, camera_id=0):
        """
        Executa inferência em tempo real com webcam
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Erro: Não foi possível abrir a câmera {camera_id}")
            return
        
        print(f"📹 Inferência em tempo real - Pressione 'q' para sair")
        
        frame_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Estima movimento
            if prev_frame is not None:
                relative_pose = self.estimate_camera_motion(frame, prev_frame)
                current_pose = self.camera_poses[-1] @ relative_pose
                self.camera_poses.append(current_pose)
            
            # Processa
            annotated_frame, tracked_detections = self.process_frame(frame, frame_count)
            valid_detections = self.validate_session_positions(
                tracked_detections, self.camera_poses[-1])
            
            if valid_detections:
                frame_height, frame_width = frame.shape[:2]
                self.estimate_3d_positions(
                    valid_detections, self.camera_poses[-1], frame_width, frame_height)
            
            self.draw_duct_info(annotated_frame, valid_detections)
            self.update_fps()
            
            cv2.imshow('Detecção de Sessões - Tempo Real', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_frame = frame.copy()
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Identificação de Sessões de Dutos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Processar vídeo básico
  python duct_detector.py -m modelo.pt -v video.mp4
  
  # Processar com saída e reconstrução 3D
  python duct_detector.py -m modelo.pt -v video.mp4 -o resultado.mp4 -r reconstrucao.json
  
  # Câmera em tempo real
  python duct_detector.py -m modelo.pt -c 0
  
  # Ajustar confiança
  python duct_detector.py -m modelo.pt -v video.mp4 --confidence 0.7
        """)
    
    parser.add_argument('--model', '-m', required=True,
                       help='Caminho para o modelo YOLO treinado (.pt)')
    parser.add_argument('--video', '-v',
                       help='Caminho para o arquivo de vídeo')
    parser.add_argument('--camera', '-c', type=int,
                       help='ID da câmera (default: 0)')
    parser.add_argument('--output', '-o',
                       help='Caminho para salvar vídeo de saída')
    parser.add_argument('--confidence', '--conf', type=float, default=0.5,
                       help='Limiar de confiança (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Desabilita exibição em tempo real')
    parser.add_argument('--reconstruction', '-r',
                       help='Caminho para salvar dados de reconstrução 3D (JSON)')
    
    args = parser.parse_args()
    
    # Validações
    if not os.path.exists(args.model):
        print(f"❌ Erro: Arquivo do modelo não encontrado: {args.model}")
        return
    
    if not args.video and args.camera is None:
        print("❌ Erro: Especifique --video ou --camera")
        parser.print_help()
        return
    
    if args.video and not os.path.exists(args.video):
        print(f"❌ Erro: Arquivo de vídeo não encontrado: {args.video}")
        return
    
    # Inicializa detector
    print("🚀 Inicializando Sistema de Detecção de Sessões de Dutos")
    detector = DuctDetectorLocal(args.model, args.confidence)
    
    # Executa inferência
    try:
        if args.video:
            detector.run_video_inference(
                args.video, args.output, args.reconstruction, not args.no_display)
        else:
            detector.run_webcam_inference(args.camera)
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()