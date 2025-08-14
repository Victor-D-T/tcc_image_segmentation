"""
Treinamento de modelo YOLO para detecção de rebites em estruturas de aeronaves
Versão simplificada focada apenas no treinamento
"""

import yaml
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
import shutil


class RivetYOLOTrainer:
    def __init__(self, data_root: str = "rivetes"):
        """
        Inicializa o treinador YOLO para rebites
        
        Args:
            data_root: Diretório raiz com as pastas images, labels, classes
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels" 
        self.classes_file = self.data_root / "classes" / "classes.txt"
        
        # Verificar estrutura de diretórios
        self.validate_data_structure()
        
        # Carregar classes
        self.classes = self.load_classes()
        
    def validate_data_structure(self):
        """Valida se a estrutura de diretórios está correta"""
        required_dirs = [self.images_dir, self.labels_dir]
        required_files = [self.classes_file]
        
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Diretório obrigatório não encontrado: {directory}")
                
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {file_path}")
        
        print(f"✅ Estrutura de dados validada em {self.data_root}")
        
    def load_classes(self):
        """Carrega as classes do arquivo"""
        with open(self.classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"📋 Classes carregadas: {classes}")
        return classes
    
    def create_dataset_yaml(self, train_split: float = 0.8):
        """
        Cria arquivo YAML de configuração do dataset e organiza dados
        
        Args:
            train_split: Proporção de dados para treinamento (0.8 = 80%)
        """
        # Criar estrutura YOLO padrão
        dataset_dir = self.data_root / "yolo_dataset"
        train_images_dir = dataset_dir / "train" / "images"
        train_labels_dir = dataset_dir / "train" / "labels"
        val_images_dir = dataset_dir / "val" / "images"
        val_labels_dir = dataset_dir / "val" / "labels"
        
        # Criar diretórios
        for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Listar todas as imagens
        image_files = list(self.images_dir.glob("*.jpg"))
        print(f"📸 Encontradas {len(image_files)} imagens")
        
        # Dividir em treino e validação
        num_train = int(len(image_files) * train_split)
        train_files = image_files[:num_train]
        val_files = image_files[num_train:]
        
        print(f"🔄 Divisão: {len(train_files)} treino, {len(val_files)} validação")
        
        # Copiar arquivos para estrutura YOLO
        self._copy_files(train_files, train_images_dir, train_labels_dir)
        self._copy_files(val_files, val_images_dir, val_labels_dir)
        
        # Criar arquivo YAML
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Dataset YAML criado: {yaml_path}")
        return yaml_path
    
    def _copy_files(self, image_files, dest_images_dir, dest_labels_dir):
        """Copia imagens e labels correspondentes"""
        for img_file in image_files:
            # Copiar imagem
            shutil.copy2(img_file, dest_images_dir / img_file.name)
            
            # Copiar label correspondente (mesmo nome, extensão .txt)
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, dest_labels_dir / f"{img_file.stem}.txt")
            else:
                print(f"⚠️  Label não encontrado para {img_file.name}")
    
    def train_model(self, yaml_path: str, epochs: int = 100, img_size: int = 640, 
                   batch_size: int = 16, model_size: str = "n"):
        """
        Treina o modelo YOLO
        
        Args:
            yaml_path: Caminho para o arquivo YAML do dataset
            epochs: Número de épocas de treinamento
            img_size: Tamanho das imagens (640x640)
            batch_size: Tamanho do batch
            model_size: Tamanho do modelo ('n', 's', 'm', 'l', 'x')
        """
        print(f"🚀 Iniciando treinamento YOLO{model_size} por {epochs} épocas")
        
        # Inicializar modelo
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Treinar
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='rivet_detection',
            save=True,
            plots=True,
            val=True,
            patience=20,  # Early stopping
            save_period=10  # Salvar checkpoint a cada 10 épocas
        )
        
        print("✅ Treinamento concluído!")
        return results
    
    def validate_model(self, model_path: str, yaml_path: str):
        """
        Valida o modelo treinado
        
        Args:
            model_path: Caminho para o modelo treinado
            yaml_path: Caminho para o arquivo YAML do dataset
        """
        print("🔍 Validando modelo...")
        
        model = YOLO(model_path)
        results = model.val(data=yaml_path)
        
        print(f"📊 Resultados da validação:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def test_inference(self, model_path: str, test_image_path: str = None):
        """
        Testa inferência em pelo menos 3 imagens e mostra resultados visualmente
        
        Args:
            model_path: Caminho para o modelo treinado
            test_image_path: Caminho para imagem específica (se None, usa 3 do dataset)
        """
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import random
        
        print("🔮 Testando inferência com visualização...")
        
        model = YOLO(model_path)
        
        # Selecionar imagens para teste
        if test_image_path is not None:
            test_images = [Path(test_image_path)]
        else:
            # Pegar pelo menos 3 imagens do dataset
            all_images = list(self.images_dir.glob("*.jpg"))
            if len(all_images) < 3:
                test_images = all_images
                print(f"⚠️  Apenas {len(all_images)} imagens disponíveis")
            else:
                test_images = random.sample(all_images, min(3, len(all_images)))
        
        print(f"📸 Testando {len(test_images)} imagens")
        
        # Configurar subplot para mostrar as imagens
        fig, axes = plt.subplots(1, len(test_images), figsize=(6*len(test_images), 6))
        if len(test_images) == 1:
            axes = [axes]
        
        all_results = []
        
        for idx, img_path in enumerate(test_images):
            print(f"\n🔍 Processando: {img_path.name}")
            
            # Fazer predição
            results = model.predict(
                source=str(img_path),
                conf=0.3,  # Confidence mais baixa para capturar mais detecções
                save=False,  # Não salvar arquivo, vamos mostrar no matplotlib
                verbose=False
            )
            
            all_results.append(results[0])
            
            # Carregar imagem
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Mostrar imagem
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"{img_path.name}", fontsize=12)
            axes[idx].axis('off')
            
            # Adicionar bounding boxes
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                print(f"🎯 Detectados {len(boxes)} objetos:")
                
                # Cores diferentes para cada classe
                colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                    color = colors[class_id % len(colors)]
                    
                    # Adicionar retângulo
                    rect = Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', alpha=0.8)
                    axes[idx].add_patch(rect)
                    
                    # Adicionar label
                    label = f"{class_name}: {conf:.2f}"
                    axes[idx].text(x1, y1-5, label, 
                                 bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor=color, alpha=0.7),
                                 fontsize=9, color='white', weight='bold')
                    
                    print(f"   {i+1}: {class_name} (conf: {conf:.2f}) - Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            else:
                print("❌ Nenhum objeto detectado")
                axes[idx].text(0.5, 0.5, 'Sem detecções', 
                             transform=axes[idx].transAxes, 
                             fontsize=14, ha='center', va='center',
                             bbox=dict(boxstyle="round", facecolor='red', alpha=0.7),
                             color='white', weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Estatísticas gerais
        total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in all_results)
        avg_conf = 0
        class_counts = {}
        
        if total_detections > 0:
            all_confs = []
            for result in all_results:
                if result.boxes is not None:
                    confs = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    all_confs.extend(confs)
                    
                    for class_id in class_ids:
                        class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            avg_conf = sum(all_confs) / len(all_confs)
        
        print(f"\n📊 Resumo das detecções:")
        print(f"   Total de detecções: {total_detections}")
        print(f"   Confiança média: {avg_conf:.3f}")
        print(f"   Por classe: {class_counts}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Treinamento YOLO para Detecção de Rebites')
    parser.add_argument('--data-root', '-d', default='rivetes', help='Diretório raiz dos dados')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Tamanho do batch')
    parser.add_argument('--img-size', '-i', type=int, default=640, help='Tamanho da imagem')
    parser.add_argument('--model-size', '-m', default='n', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Tamanho do modelo YOLO')
    parser.add_argument('--train-split', '-s', type=float, default=0.8, 
                       help='Proporção de dados para treinamento')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Apenas validar modelo existente')
    parser.add_argument('--model-path', help='Caminho para modelo existente (para validação)')
    parser.add_argument('--test-image', help='Imagem para teste de inferência')
    
    args = parser.parse_args()
    
    try:
        # Inicializar treinador
        trainer = RivetYOLOTrainer(args.data_root)
        
        if args.validate_only:
            if not args.model_path:
                print("❌ Modelo não especificado para validação")
                return
            
            # Criar yaml para validação
            yaml_path = trainer.create_dataset_yaml(args.train_split)
            
            # Validar modelo
            trainer.validate_model(args.model_path, yaml_path)
            
            # Teste de inferência
            trainer.test_inference(args.model_path, args.test_image)
            
        else:
            # Preparar dataset
            yaml_path = trainer.create_dataset_yaml(args.train_split)
            
            # Treinar modelo
            results = trainer.train_model(
                yaml_path=yaml_path,
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch_size,
                model_size=args.model_size
            )
            
            # Modelo treinado fica em runs/detect/rivet_detection/weights/best.pt
            best_model = "runs/detect/rivet_detection/weights/best.pt"
            
            if os.path.exists(best_model):
                print(f"🎉 Modelo salvo em: {best_model}")
                
                # Teste de inferência
                trainer.test_inference(best_model, args.test_image)
            else:
                print("❌ Modelo não encontrado após treinamento")
    
    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()