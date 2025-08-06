"""
메모리 모니터링 유틸리티
K-Fold 학습 중 메모리 사용량을 모니터링하고 정리
"""

import torch
import gc
import psutil
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """메모리 사용량 모니터링 및 정리"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_info()

    def get_memory_info(self) -> Dict[str, float]:
        """현재 메모리 사용량 정보 반환"""
        info = {}

        # 시스템 RAM
        memory_info = self.process.memory_info()
        info['ram_used_gb'] = memory_info.rss / 1024**3
        info['ram_percent'] = self.process.memory_percent()

        # GPU 메모리
        if torch.cuda.is_available():
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_cached_gb'] = torch.cuda.memory_reserved() / 1024**3
            info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['gpu_free_gb'] = info['gpu_total_gb'] - info['gpu_allocated_gb']
        else:
            info['gpu_allocated_gb'] = 0
            info['gpu_cached_gb'] = 0
            info['gpu_total_gb'] = 0
            info['gpu_free_gb'] = 0

        return info

    def print_memory_status(self, prefix: str = ""):
        """메모리 상태 출력"""
        info = self.get_memory_info()

        print(f"\n📊 {prefix} 메모리 상태:")
        print(f"   💾 RAM: {info['ram_used_gb']:.2f}GB ({info['ram_percent']:.1f}%)")

        if torch.cuda.is_available():
            print(f"   🎮 GPU: {info['gpu_allocated_gb']:.2f}GB / {info['gpu_total_gb']:.2f}GB")
            print(f"   📦 GPU 캐시: {info['gpu_cached_gb']:.2f}GB")
            print(f"   🆓 GPU 여유: {info['gpu_free_gb']:.2f}GB")

            # 메모리 부족 경고
            if info['gpu_free_gb'] < 1.0:
                print(f"   ⚠️ GPU 메모리 부족 경고! (여유: {info['gpu_free_gb']:.2f}GB)")

    def cleanup_memory(self, aggressive: bool = False):
        """메모리 정리"""
        logger.info("🧹 메모리 정리 시작...")

        # Python 가비지 컬렉션
        collected = gc.collect()
        logger.info(f"   Python GC: {collected}개 객체 정리")

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if aggressive:
                # 더 적극적인 GPU 메모리 정리
                torch.cuda.ipc_collect()

            logger.info("   GPU 캐시 정리 완료")

        # 메모리 상태 출력
        self.print_memory_status("정리 후")

    def check_memory_safety(self, min_gpu_gb: float = 2.0, min_ram_percent: float = 80.0) -> bool:
        """메모리 안전성 확인"""
        info = self.get_memory_info()

        # GPU 메모리 확인
        if torch.cuda.is_available() and info['gpu_free_gb'] < min_gpu_gb:
            logger.warning(f"GPU 메모리 부족: {info['gpu_free_gb']:.2f}GB < {min_gpu_gb}GB")
            return False

        # RAM 사용률 확인
        if info['ram_percent'] > min_ram_percent:
            logger.warning(f"RAM 사용률 높음: {info['ram_percent']:.1f}% > {min_ram_percent}%")
            return False

        return True

    def wait_for_memory_recovery(self, max_wait_seconds: int = 30):
        """메모리 회복 대기"""
        import time

        logger.info("⏳ 메모리 회복 대기 중...")

        for i in range(max_wait_seconds):
            if self.check_memory_safety():
                logger.info(f"✅ 메모리 회복 완료 ({i+1}초 대기)")
                return True

            time.sleep(1)

            # 5초마다 정리 시도
            if (i + 1) % 5 == 0:
                self.cleanup_memory(aggressive=True)

        logger.warning(f"⚠️ {max_wait_seconds}초 대기 후에도 메모리 부족")
        return False


def safe_fold_training_wrapper(fold_training_func):
    """K-Fold 학습을 메모리 안전하게 래핑하는 데코레이터"""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        fold_num = args[1] if len(args) > 1 else kwargs.get('fold_num', 'unknown')

        try:
            # 학습 전 메모리 상태 확인
            monitor.print_memory_status(f"Fold {fold_num} 학습 전")

            # 메모리 안전성 확인
            if not monitor.check_memory_safety():
                logger.warning(f"Fold {fold_num} 시작 전 메모리 부족, 정리 시도...")
                monitor.cleanup_memory(aggressive=True)
                monitor.wait_for_memory_recovery()

            # 실제 학습 실행
            result = fold_training_func(*args, **kwargs)

            # 학습 후 메모리 정리
            monitor.print_memory_status(f"Fold {fold_num} 학습 후")
            monitor.cleanup_memory(aggressive=True)

            return result

        except Exception as e:
            logger.error(f"Fold {fold_num} 학습 중 오류: {e}")
            # 오류 발생 시에도 메모리 정리
            monitor.cleanup_memory(aggressive=True)
            raise

    return wrapper


# 전역 메모리 모니터
global_monitor = MemoryMonitor()


def print_memory_status(prefix: str = ""):
    """전역 메모리 상태 출력"""
    global_monitor.print_memory_status(prefix)


def cleanup_memory(aggressive: bool = False):
    """전역 메모리 정리"""
    global_monitor.cleanup_memory(aggressive)


def check_memory_safety(min_gpu_gb: float = 2.0) -> bool:
    """전역 메모리 안전성 확인"""
    return global_monitor.check_memory_safety(min_gpu_gb)


if __name__ == "__main__":
    # 테스트
    monitor = MemoryMonitor()
    monitor.print_memory_status("테스트")
    monitor.cleanup_memory()
    print("✅ 메모리 모니터 테스트 완료")
