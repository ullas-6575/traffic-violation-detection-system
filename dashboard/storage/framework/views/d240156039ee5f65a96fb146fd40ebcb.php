<?php if (isset($component)) { $__componentOriginal9ac128a9029c0e4701924bd2d73d7f54 = $component; } ?>
<?php if (isset($attributes)) { $__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54 = $attributes; } ?>
<?php $component = App\View\Components\AppLayout::resolve([] + (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag ? $attributes->all() : [])); ?>
<?php $component->withName('app-layout'); ?>
<?php if ($component->shouldRender()): ?>
<?php $__env->startComponent($component->resolveView(), $component->data()); ?>
<?php if (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag): ?>
<?php $attributes = $attributes->except(\App\View\Components\AppLayout::ignoredParameterNames()); ?>
<?php endif; ?>
<?php $component->withAttributes([]); ?>
     <?php $__env->slot('header', null, []); ?> 
        <div class="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[0.18em] text-red-600">Evidence register</p>
                <h2 class="text-2xl font-bold leading-tight text-slate-900">Traffic violations</h2>
            </div>
            <p class="text-sm text-slate-500">Newest evidence appears first</p>
        </div>
     <?php $__env->endSlot(); ?>

    <div class="py-8">
        <div class="mx-auto max-w-7xl space-y-6 px-4 sm:px-6 lg:px-8">
            <section class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Violation summary">
                <?php $__currentLoopData = [
                    ['label' => 'Total records', 'value' => $summary['total'], 'tone' => 'text-slate-900'],
                    ['label' => 'Overspeed', 'value' => $summary['overspeed'], 'tone' => 'text-amber-700'],
                    ['label' => 'Red light', 'value' => $summary['red_light'], 'tone' => 'text-red-700'],
                    ['label' => 'Pending review', 'value' => $summary['pending'], 'tone' => 'text-indigo-700'],
                ]; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as $card): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); ?>
                    <article class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                        <p class="text-sm font-medium text-slate-500"><?php echo e($card['label']); ?></p>
                        <p class="mt-2 text-3xl font-bold <?php echo e($card['tone']); ?>"><?php echo e(number_format($card['value'])); ?></p>
                    </article>
                <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); ?>
            </section>

            <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <form method="GET" action="<?php echo e(route('violations.index')); ?>" class="grid gap-4 lg:grid-cols-6">
                    <div class="lg:col-span-2">
                        <label for="search" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Plate or event ID</label>
                        <input id="search" name="search" value="<?php echo e(request('search')); ?>" placeholder="Search evidence…"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div>
                        <label for="type" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Violation</label>
                        <select id="type" name="type" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All types</option>
                            <option value="OVERSPEED" <?php if(request('type') === 'OVERSPEED'): echo 'selected'; endif; ?>>Overspeed</option>
                            <option value="RED_LIGHT" <?php if(request('type') === 'RED_LIGHT'): echo 'selected'; endif; ?>>Red light</option>
                        </select>
                    </div>
                    <div>
                        <label for="color" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Color</label>
                        <select id="color" name="color" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All colors</option>
                            <?php $__currentLoopData = $colors; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as $color): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); ?>
                                <option value="<?php echo e($color); ?>" <?php if(request('color') === $color): echo 'selected'; endif; ?>><?php echo e(ucfirst(strtolower($color))); ?></option>
                            <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); ?>
                        </select>
                    </div>
                    <div>
                        <label for="status" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Status</label>
                        <select id="status" name="status" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All statuses</option>
                            <?php $__currentLoopData = ['PENDING', 'CONFIRMED', 'DISMISSED']; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as $status): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); ?>
                                <option value="<?php echo e($status); ?>" <?php if(request('status') === $status): echo 'selected'; endif; ?>><?php echo e(ucfirst(strtolower($status))); ?></option>
                            <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); ?>
                        </select>
                    </div>
                    <div>
                        <label for="date_from" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">From</label>
                        <input id="date_from" type="date" name="date_from" value="<?php echo e(request('date_from')); ?>"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div>
                        <label for="date_to" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">To</label>
                        <input id="date_to" type="date" name="date_to" value="<?php echo e(request('date_to')); ?>"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div class="flex items-end gap-2 lg:col-span-6">
                        <button class="rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-slate-700">Apply filters</button>
                        <a href="<?php echo e(route('violations.index')); ?>" class="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50">Reset</a>
                    </div>
                </form>
            </section>

            <section class="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-slate-200">
                        <thead class="bg-slate-50">
                            <tr class="text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                                <th class="px-5 py-3">Evidence</th>
                                <th class="px-5 py-3">Plate</th>
                                <th class="px-5 py-3">Violation</th>
                                <th class="px-5 py-3">Speed</th>
                                <th class="px-5 py-3">Vehicle</th>
                                <th class="px-5 py-3">Confidence</th>
                                <th class="px-5 py-3">Status</th>
                                <th class="px-5 py-3">Captured</th>
                                <th class="px-5 py-3"><span class="sr-only">Actions</span></th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 bg-white">
                            <?php $__empty_1 = true; $__currentLoopData = $violations; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as $violation): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); $__empty_1 = false; ?>
                                <tr class="transition hover:bg-slate-50/80">
                                    <td class="px-5 py-4">
                                        <?php if($violation->image_path): ?>
                                            <img src="<?php echo e(asset('storage/'.$violation->image_path)); ?>" alt="Evidence for <?php echo e($violation->event_id); ?>"
                                                 class="h-16 w-24 rounded-lg bg-slate-100 object-cover ring-1 ring-slate-200">
                                        <?php else: ?>
                                            <div class="flex h-16 w-24 items-center justify-center rounded-lg bg-slate-100 text-xs font-medium text-slate-400">No image</div>
                                        <?php endif; ?>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <p class="font-semibold <?php echo e($violation->plate_number === 'UNREADABLE' ? 'text-red-700' : 'text-slate-900'); ?>"><?php echo e($violation->plate_number); ?></p>
                                        <p class="mt-1 font-mono text-xs text-slate-400"><?php echo e($violation->event_id); ?></p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold <?php echo e($violation->violation_type === 'RED_LIGHT' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'); ?>">
                                            <?php echo e(str_replace('_', ' ', $violation->violation_type)); ?>

                                        </span>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <?php if($violation->speed !== null): ?>
                                            <strong><?php echo e(number_format($violation->speed, 1)); ?></strong> km/h
                                            <?php if($violation->speed_limit): ?>
                                                <p class="text-xs text-slate-400">Limit <?php echo e(number_format($violation->speed_limit, 0)); ?></p>
                                            <?php endif; ?>
                                        <?php else: ?>
                                            <span class="text-slate-400">—</span>
                                        <?php endif; ?>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <p><?php echo e(ucfirst(strtolower($violation->vehicle_color))); ?></p>
                                        <p class="text-xs text-slate-400"><?php echo e($violation->direction ?: 'Unknown'); ?></p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <?php echo e(number_format(($violation->ocr_confidence ?? 0) * 100, 1)); ?>%
                                        <p class="text-xs text-slate-400"><?php echo e($violation->ocr_engine ?: 'none'); ?></p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold <?php echo e(match($violation->status) {
                                            'CONFIRMED' => 'bg-emerald-100 text-emerald-800',
                                            'DISMISSED' => 'bg-slate-200 text-slate-700',
                                            default => 'bg-indigo-100 text-indigo-800',
                                        }); ?>"><?php echo e(ucfirst(strtolower($violation->status))); ?></span>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <time datetime="<?php echo e($violation->frame_timestamp?->toIso8601String()); ?>">
                                            <?php echo e($violation->frame_timestamp?->format('d M Y')); ?>

                                            <span class="block text-xs text-slate-400"><?php echo e($violation->frame_timestamp?->format('H:i:s')); ?></span>
                                        </time>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-right">
                                        <a href="<?php echo e(route('violations.show', $violation)); ?>" class="font-semibold text-red-700 hover:text-red-900">Review</a>
                                    </td>
                                </tr>
                            <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); if ($__empty_1): ?>
                                <tr>
                                    <td colspan="9" class="px-6 py-16 text-center">
                                        <p class="font-semibold text-slate-700">No violations match these filters.</p>
                                        <p class="mt-1 text-sm text-slate-500">Try clearing one or more search fields.</p>
                                    </td>
                                </tr>
                            <?php endif; ?>
                        </tbody>
                    </table>
                </div>
                <?php if($violations->hasPages()): ?>
                    <div class="border-t border-slate-200 px-5 py-4"><?php echo e($violations->links()); ?></div>
                <?php endif; ?>
            </section>
        </div>
    </div>
 <?php echo $__env->renderComponent(); ?>
<?php endif; ?>
<?php if (isset($__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54)): ?>
<?php $attributes = $__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54; ?>
<?php unset($__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54); ?>
<?php endif; ?>
<?php if (isset($__componentOriginal9ac128a9029c0e4701924bd2d73d7f54)): ?>
<?php $component = $__componentOriginal9ac128a9029c0e4701924bd2d73d7f54; ?>
<?php unset($__componentOriginal9ac128a9029c0e4701924bd2d73d7f54); ?>
<?php endif; ?>
<?php /**PATH E:\ISD\app\resources\views/violations/index.blade.php ENDPATH**/ ?>