<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[0.18em] text-red-600">Evidence register</p>
                <h2 class="text-2xl font-bold leading-tight text-slate-900">Traffic violations</h2>
            </div>
            <p class="text-sm text-slate-500">Newest evidence appears first</p>
        </div>
    </x-slot>

    <div class="py-8">
        <div class="mx-auto max-w-7xl space-y-6 px-4 sm:px-6 lg:px-8">
            <section class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Violation summary">
                @foreach ([
                    ['label' => 'Total records', 'value' => $summary['total'], 'tone' => 'text-slate-900'],
                    ['label' => 'Overspeed', 'value' => $summary['overspeed'], 'tone' => 'text-amber-700'],
                    ['label' => 'Red light', 'value' => $summary['red_light'], 'tone' => 'text-red-700'],
                    ['label' => 'Pending review', 'value' => $summary['pending'], 'tone' => 'text-indigo-700'],
                ] as $card)
                    <article class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                        <p class="text-sm font-medium text-slate-500">{{ $card['label'] }}</p>
                        <p class="mt-2 text-3xl font-bold {{ $card['tone'] }}">{{ number_format($card['value']) }}</p>
                    </article>
                @endforeach
            </section>

            <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <form method="GET" action="{{ route('violations.index') }}" class="grid gap-4 lg:grid-cols-6">
                    <div class="lg:col-span-2">
                        <label for="search" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Plate or event ID</label>
                        <input id="search" name="search" value="{{ request('search') }}" placeholder="Search evidence…"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div>
                        <label for="type" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Violation</label>
                        <select id="type" name="type" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All types</option>
                            <option value="OVERSPEED" @selected(request('type') === 'OVERSPEED')>Overspeed</option>
                            <option value="RED_LIGHT" @selected(request('type') === 'RED_LIGHT')>Red light</option>
                        </select>
                    </div>
                    <div>
                        <label for="color" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Color</label>
                        <select id="color" name="color" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All colors</option>
                            @foreach ($colors as $color)
                                <option value="{{ $color }}" @selected(request('color') === $color)>{{ ucfirst(strtolower($color)) }}</option>
                            @endforeach
                        </select>
                    </div>
                    <div>
                        <label for="status" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">Status</label>
                        <select id="status" name="status" class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                            <option value="">All statuses</option>
                            @foreach (['PENDING', 'CONFIRMED', 'DISMISSED'] as $status)
                                <option value="{{ $status }}" @selected(request('status') === $status)>{{ ucfirst(strtolower($status)) }}</option>
                            @endforeach
                        </select>
                    </div>
                    <div>
                        <label for="date_from" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">From</label>
                        <input id="date_from" type="date" name="date_from" value="{{ request('date_from') }}"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div>
                        <label for="date_to" class="block text-xs font-semibold uppercase tracking-wide text-slate-600">To</label>
                        <input id="date_to" type="date" name="date_to" value="{{ request('date_to') }}"
                               class="mt-1 block w-full rounded-lg border-slate-300 text-sm shadow-sm focus:border-red-500 focus:ring-red-500">
                    </div>
                    <div class="flex items-end gap-2 lg:col-span-6">
                        <button class="rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-slate-700">Apply filters</button>
                        <a href="{{ route('violations.index') }}" class="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50">Reset</a>
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
                            @forelse ($violations as $violation)
                                <tr class="transition hover:bg-slate-50/80">
                                    <td class="px-5 py-4">
                                        @if ($violation->image_path)
                                            <img src="{{ asset('storage/'.$violation->image_path) }}" alt="Evidence for {{ $violation->event_id }}"
                                                 class="h-16 w-24 rounded-lg bg-slate-100 object-cover ring-1 ring-slate-200">
                                        @else
                                            <div class="flex h-16 w-24 items-center justify-center rounded-lg bg-slate-100 text-xs font-medium text-slate-400">No image</div>
                                        @endif
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <p class="font-semibold {{ $violation->plate_number === 'UNREADABLE' ? 'text-red-700' : 'text-slate-900' }}">{{ $violation->plate_number }}</p>
                                        <p class="mt-1 font-mono text-xs text-slate-400">{{ $violation->event_id }}</p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold {{ $violation->violation_type === 'RED_LIGHT' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800' }}">
                                            {{ str_replace('_', ' ', $violation->violation_type) }}
                                        </span>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        @if ($violation->speed !== null)
                                            <strong>{{ number_format($violation->speed, 1) }}</strong> km/h
                                            @if ($violation->speed_limit)
                                                <p class="text-xs text-slate-400">Limit {{ number_format($violation->speed_limit, 0) }}</p>
                                            @endif
                                        @else
                                            <span class="text-slate-400">—</span>
                                        @endif
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <p>{{ ucfirst(strtolower($violation->vehicle_color)) }}</p>
                                        <p class="text-xs text-slate-400">{{ $violation->direction ?: 'Unknown' }}</p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        {{ number_format(($violation->ocr_confidence ?? 0) * 100, 1) }}%
                                        <p class="text-xs text-slate-400">{{ $violation->ocr_engine ?: 'none' }}</p>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4">
                                        <span class="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold {{ match($violation->status) {
                                            'CONFIRMED' => 'bg-emerald-100 text-emerald-800',
                                            'DISMISSED' => 'bg-slate-200 text-slate-700',
                                            default => 'bg-indigo-100 text-indigo-800',
                                        } }}">{{ ucfirst(strtolower($violation->status)) }}</span>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                                        <time datetime="{{ $violation->frame_timestamp?->toIso8601String() }}">
                                            {{ $violation->frame_timestamp?->format('d M Y') }}
                                            <span class="block text-xs text-slate-400">{{ $violation->frame_timestamp?->format('H:i:s') }}</span>
                                        </time>
                                    </td>
                                    <td class="whitespace-nowrap px-5 py-4 text-right">
                                        <a href="{{ route('violations.show', $violation) }}" class="font-semibold text-red-700 hover:text-red-900">Review</a>
                                    </td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="9" class="px-6 py-16 text-center">
                                        <p class="font-semibold text-slate-700">No violations match these filters.</p>
                                        <p class="mt-1 text-sm text-slate-500">Try clearing one or more search fields.</p>
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
                @if ($violations->hasPages())
                    <div class="border-t border-slate-200 px-5 py-4">{{ $violations->links() }}</div>
                @endif
            </section>
        </div>
    </div>
</x-app-layout>
